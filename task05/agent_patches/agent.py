import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from arc_agi import EnvironmentWrapper
from arc_agi.scorecard import EnvironmentScorecard
from arcengine import FrameData, FrameDataRaw, GameAction, GameState
from pydantic import ValidationError

from .recorder import Recorder
from .tracing import trace_agent_session
from .gif_recorder import GIFRecorder

logger = logging.getLogger()


class Agent(ABC):
    """Interface for an agent that plays one ARC-AGI-3 game."""

    MAX_ACTIONS: int = 80  # to avoid looping forever if agent doesnt exit
    ROOT_URL: str

    action_counter: int = 0

    timer: float = 0
    agent_name: str
    card_id: str
    game_id: str
    guid: str
    frames: list[FrameData]

    recorder: Recorder
    headers: dict[str, str]
    arc_env: EnvironmentWrapper

    # AgentOps tracing attributes
    trace: Any = None
    tags: list[str]

    def __init__(
        self,
        card_id: str,
        game_id: str,
        agent_name: str,
        ROOT_URL: str,
        record: bool,
        arc_env: EnvironmentWrapper,
        tags: Optional[list[str]] = None,
    ) -> None:
        self.ROOT_URL = ROOT_URL
        self.card_id = card_id
        self.game_id = game_id
        self.guid = ""
        self.agent_name = agent_name
        self.tags = tags or []
        self.frames = [FrameData(levels_completed=0)]
        self._cleanup = True
        if record:
            self.start_recording()
        self.headers = {
            "X-API-Key": os.getenv("ARC_API_KEY", ""),
            "Accept": "application/json",
        }
        self.arc_env = arc_env
        self.gif_recorder = GIFRecorder(game_id=self.game_id, agent_name=self.agent_name)

    @trace_agent_session
    def main(self) -> None:
        """The main agent loop. Play the game_id until finished, then exits."""
        self.timer = time.time()
        try:
            if self.arc_env and hasattr(self.arc_env, "observation_space") and self.arc_env.observation_space:
                raw_init = self._convert_raw_frame_data(self.arc_env.observation_space)
                if raw_init.frame:
                    self.gif_recorder.record_frame(raw_init.frame[-1])
        except Exception as e:
            logger.warning(f"Failed to record initial frame: {e}")

        while (
            not self.is_done(self.frames, self.frames[-1])
            and self.action_counter <= self.MAX_ACTIONS
        ):
            action = self.choose_action(
                self.frames,
                self._convert_raw_frame_data(
                    self.arc_env.observation_space if self.arc_env else None
                ),
            )
            if frame := self.take_action(action):
                self.append_frame(frame)
                logger.info(
                    f"{self.game_id} - {action.name}: count {self.action_counter}, levels completed {frame.levels_completed}, avg fps {self.fps})"
                )
            self.action_counter += 1

        self.cleanup()

    @property
    def state(self) -> GameState:
        return self.frames[-1].state

    @property
    def levels_completed(self) -> int:
        return self.frames[-1].levels_completed  # type: ignore[no-any-return]

    @property
    def seconds(self) -> float:
        return (time.time() - self.timer) * 100 // 1 / 100

    @property
    def fps(self) -> float:
        if self.action_counter == 0:
            return 0.0
        elapsed_time = max(self.seconds, 0.1)
        return round(self.action_counter / elapsed_time, 2)

    @property
    def is_playback(self) -> bool:
        return type(self) is Playback

    @property
    def name(self) -> str:
        n = self.__class__.__name__.lower()
        return f"{self.game_id}.{n}"

    def start_recording(self) -> None:
        filename = self.agent_name if self.is_playback else None
        self.recorder = Recorder(prefix=self.name, filename=filename)
        logger.info(
            f"created new recording for {self.name} into {self.recorder.filename}"
        )

    def append_frame(self, frame: FrameData) -> None:
        self.frames.append(frame)
        if frame.guid:
            self.guid = frame.guid
        if hasattr(self, "recorder") and not self.is_playback:
            self.recorder.record(json.loads(frame.model_dump_json()))

    def do_action_request(self, action: GameAction) -> FrameData:
        data = action.action_data.model_dump()
        
        # 進行點擊座標的自動對齊修正 (避免 LLM 座標偏誤或網格-像素座標混淆)
        if action.name == "ACTION6" and "x" in data and "y" in data:
            try:
                latest_frame = self.frames[-1]
                if latest_frame and latest_frame.frame:
                    import numpy as np
                    grid = np.array(latest_frame.frame[-1])
                    h, w = grid.shape
                    
                    scale_x = int(64 / w)
                    scale_y = int(64 / h)
                    scale = min(scale_x, scale_y)
                    x_padding = int((64 - (w * scale)) / 2)
                    y_padding = int((64 - (h * scale)) / 2)
                    
                    raw_x = data["x"]
                    raw_y = data["y"]
                    
                    if raw_x < w and raw_y < h:
                        target_grid_x = raw_x
                        target_grid_y = raw_y
                    else:
                        target_grid_x = int((raw_x - x_padding) / scale) if raw_x - x_padding >= 0 else -1
                        target_grid_y = int((raw_y - y_padding) / scale) if raw_y - y_padding >= 0 else -1
                    
                    btn_coords = []
                    for r in range(h):
                        for c in range(w):
                            if grid[r, c] in [1, 9]:
                                btn_coords.append((c, r))
                                
                    if btn_coords:
                        best_btn = None
                        min_dist = float('inf')
                        for bc_x, bc_y in btn_coords:
                            dist = (bc_x - target_grid_x)**2 + (bc_y - target_grid_y)**2
                            if dist < min_dist:
                                min_dist = dist
                                best_btn = (bc_x, bc_y)
                        
                        # Only snap if the click is reasonably close to a button (dist_sq <= 8)
                        if best_btn and min_dist <= 8:
                            btn_gx, btn_gy = best_btn
                            display_x = x_padding + int((btn_gx + 0.5) * scale)
                            display_y = y_padding + int((btn_gy + 0.5) * scale)
                            
                            data["x"] = display_x
                            data["y"] = display_y
                            logger.info(f"[ACTION6 Alignment] Aligned raw click ({raw_x}, {raw_y}) to button at grid ({btn_gx}, {btn_gy}) -> display ({display_x}, {display_y})")
                        else:
                            logger.info(f"[ACTION6 Alignment] Click ({raw_x}, {raw_y}) is too far from any button (min_dist_sq={min_dist} > 8). No alignment applied.")
            except Exception as e:
                logger.warning(f"Failed to auto-align ACTION6 coordinates: {e}")

        # Agents attach reasoning to the GameAction enum instance, not to
        # action_data — read it from the enum so it actually reaches step().
        # Some agents emit a bare string; wrap so the wrapper always sees a dict.
        reasoning = getattr(action, "reasoning", None)
        if reasoning is not None and not isinstance(reasoning, dict):
            reasoning = {"text": str(reasoning)}
        raw = self.arc_env.step(action, data=data, reasoning=reasoning)
        return self._convert_raw_frame_data(raw)

    def _convert_raw_frame_data(self, raw: FrameDataRaw | None) -> FrameData:
        if raw is None:
            raise ValueError("Received None frame data from environment")
        out = FrameData(
            game_id=raw.game_id,
            frame=[arr.tolist() for arr in raw.frame],
            state=raw.state,
            levels_completed=raw.levels_completed,
            win_levels=raw.win_levels,
            guid=raw.guid,
            full_reset=raw.full_reset,
            available_actions=raw.available_actions,
        )
        return out

    def take_action(self, action: GameAction) -> Optional[FrameData]:
        """Submits the specific action and gets the next frame."""
        frame_data = self.do_action_request(action)
        try:
            frame = FrameData.model_validate(frame_data)
        except ValidationError as e:
            logger.warning(f"Incoming frame data did not validate: {e}")
            return None
        try:
            if frame and frame.frame:
                self.gif_recorder.record_frame(frame.frame[-1])
        except Exception as e:
            logger.warning(f"Failed to record step frame: {e}")
        return frame

    def cleanup(self, scorecard: Optional[EnvironmentScorecard] = None) -> None:
        """Called after main loop is finished."""
        if self._cleanup:
            self._cleanup = False  # only cleanup once per agent
            try:
                self.gif_recorder.save_gif()
            except Exception as e:
                logger.warning(f"Failed to save GIF animation: {e}")
            if hasattr(self, "recorder") and not self.is_playback:
                if scorecard:
                    self.recorder.record(scorecard.get(self.game_id))
                logger.info(
                    f"recording for {self.name} is available in {self.recorder.filename}"
                )
            if self.action_counter >= self.MAX_ACTIONS:
                logger.info(
                    f"Exiting: agent reached MAX_ACTIONS of {self.MAX_ACTIONS}, took {self.seconds} seconds ({self.fps} average fps)"
                )
            else:
                logger.info(
                    f"Finishing: agent took {self.action_counter} actions, took {self.seconds} seconds ({self.fps} average fps)"
                )

    @abstractmethod
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing or not."""
        raise NotImplementedError

    @abstractmethod
    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Choose which action the Agent should take, fill in any arguments, and return it."""
        raise NotImplementedError


class Playback(Agent):
    """An agent that plays back from a recorded session from another agent."""

    MAX_ACTIONS = 1000000
    PLAYBACK_FPS = 5

    recorded_actions: list[dict[str, Any]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.recorder = Recorder(
            prefix=Recorder.get_prefix(self.agent_name),
            guid=Recorder.get_guid(self.agent_name),
        )
        self.recorded_actions = []
        if self.agent_name in Recorder.list():
            try:
                self.recorded_actions = self.filter_actions()
                logger.info(
                    f"Loaded {len(self.recorded_actions)} actions from {self.agent_name}"
                )
            except Exception as e:
                logger.error(f"Failed to load recording {self.agent_name}: {e}")
                self.recorded_actions = []
        else:
            logger.warning(
                f"Recording {self.agent_name} not found in available recordings"
            )

    def filter_actions(self) -> list[dict[str, Any]]:
        return [
            a
            for a in self.recorder.get()
            if "data" in a and "action_input" in a["data"]
        ]

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return bool(self.action_counter >= len(self.recorded_actions))

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        loop_start_time = time.time()

        if self.action_counter >= len(self.recorded_actions):
            logger.warning(
                f"No more recorded actions available (counter: {self.action_counter}, total: {len(self.recorded_actions)})"
            )
            return GameAction.RESET

        recorded_data = self.recorded_actions[self.action_counter]["data"]
        action_input = recorded_data["action_input"]

        action = GameAction.from_id(action_input["id"])
        data = action_input["data"].copy()
        data["game_id"] = self.game_id
        action.set_data(data)
        if "reasoning" in action_input and action_input["reasoning"] is not None:
            action.reasoning = action_input["reasoning"]

        logger.debug(
            f"Playback action {self.action_counter}: {action.name} with data {data}"
        )

        target_frame_time = 1.0 / getattr(self, "PLAYBACK_FPS", 5)
        elapsed_time = time.time() - loop_start_time
        sleep_time = max(0, target_frame_time - elapsed_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

        return action

    def append_frame(self, frame: FrameData) -> None:
        # overwrite append_frame to not double record
        self.frames.append(frame)
        if frame.guid:
            self.guid = frame.guid
