"""Your ARC-AGI-3 agent.
Implemented using Google's Gemma 4 Multimodal model with Visual + Text Hybrid state representation and In-Context Learning (ICL).
"""
from __future__ import annotations

import random
import time
import os
import sys
import re
import json
import hashlib
import logging
import traceback
import math
from collections import deque
from typing import Any, Optional

# Add parent directory, task05 directory, and the user's custom venv path to resolve dependencies
sys.path.append("/home/tim9510019/wisdomCoreV2")
sys.path.append("/home/tim9510019/wisdomCoreV2/task05")
sys.path.append("/home/tim9510019/venv/lib/python3.12/site-packages")

import numpy as np
import torch
from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent

def grid_to_text_representation(grid) -> str:
    """Converts a grid to a full text representation with numpy-only connected component segment analysis."""
    grid = np.array(grid)
    H, W = grid.shape
    lines = [f"Grid size: {H} rows x {W} columns"]
    lines.append(f"Coordinate system: col=x (0=left), row=y (0=top). ACTION6(x=col, y=row) to click.")

    # 1. Detect background = most frequent color in the grid
    unique_colors, counts = np.unique(grid, return_counts=True)
    color_freq = dict(zip(unique_colors.tolist(), counts.tolist()))
    bg_color = int(unique_colors[np.argmax(counts)])
    lines.append(f"Background color (most frequent={color_freq[bg_color]} cells): Color {bg_color} — ignore for interaction.")

    # 2. Extract Connected Components via a pure Python BFS 8-connectivity algorithm
    visited = np.zeros((H, W), dtype=bool)
    objects = []
    
    # Directions for 8-connectivity
    dy = [-1, -1, -1,  0, 0,  1, 1, 1]
    dx = [-1,  0,  1, -1, 1, -1, 0, 1]

    for y in range(H):
        for x in range(W):
            val = int(grid[y, x])
            if val != bg_color and not visited[y, x]:
                # Start BFS to discover component
                component = []
                queue = [(y, x)]
                visited[y, x] = True
                
                head = 0
                while head < len(queue):
                    cy, cx = queue[head]
                    head += 1
                    component.append((cy, cx))
                    
                    for i in range(8):
                        ny, nx = cy + dy[i], cx + dx[i]
                        if 0 <= ny < H and 0 <= nx < W:
                            if int(grid[ny, nx]) == val and not visited[ny, nx]:
                                visited[ny, nx] = True
                                queue.append((ny, nx))
                
                # Compute component characteristics
                ys = [pt[0] for pt in component]
                xs = [pt[1] for pt in component]
                min_y, max_y = min(ys), max(ys)
                min_x, max_x = min(xs), max(xs)
                h_size = max_y - min_y + 1
                w_size = max_x - min_x + 1
                
                # Categorize shape structure general class
                if len(component) == 1:
                    shape_type = "Single Cell"
                elif h_size == 1 and w_size > 1:
                    shape_type = "Horizontal Linear Slider/Line"
                elif w_size == 1 and h_size > 1:
                    shape_type = "Vertical Linear Slider/Line"
                elif h_size == w_size:
                    shape_type = "Square Cluster / Dial Dial"
                else:
                    shape_type = "Complex Connected Shape"

                objects.append({
                    "color": val,
                    "size": len(component),
                    "bbox": (min_y, max_y, min_x, max_x),
                    "shape": shape_type,
                    "repr": component[:4] # show first few points
                })

    # Sort objects by Y-coordinate descending, then size descending (general order)
    objects.sort(key=lambda o: (o["bbox"][0], -o["size"]))

    if objects:
        lines.append(f"\n=== DETECTED GEOMETRIC OBJECTS ({len(objects)} distinct structures) ===")
        for idx, obj in enumerate(objects, 1):
            bbox_str = f"Rows {obj['bbox'][0]}-{obj['bbox'][1]}, Cols {obj['bbox'][2]}-{obj['bbox'][3]}"
            repr_str = ", ".join(f"(row={r},col={c})" for r, c in obj["repr"])
            lines.append(
                f" - Object {idx:02d}: Color {obj['color']:2d} | Size: {obj['size']:3d} cells | "
                f"BBox: {bbox_str} | Layout Type: {obj['shape']} | Sample Points: {repr_str}"
            )
        lines.append("=====================================================\n")
    else:
        lines.append("No non-background objects found.")

    # 3. FULL 2D map — always show all H rows for complete spatial context
    def format_val(v):
        if v == bg_color:
            return "\u00b7"  # middle dot for background
        if 0 <= v <= 9:
            return str(v)
        return chr(ord('A') + (v - 10))

    lines.append(f"Full 2D map (\u00b7=Color {bg_color} background, digits/letters=other colors):")
    for y in range(H):
        row_str = "".join(format_val(int(grid[y, x])) for x in range(W))
        lines.append(f"  Row{y:02d}: {row_str}")

    return "\n".join(lines)


# ==========================================
# 2. Global Model Caching, LoRA Wrapping & Online SFT
# ==========================================
_MODEL_CACHE = {}

class LoRALinear(torch.nn.Module):
    def __init__(self, base_linear, r=4, alpha=8):
        super().__init__()
        self.base_linear = base_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        in_features = base_linear.in_features
        out_features = base_linear.out_features
        
        # Define trainable LoRA weights
        self.lora_A = torch.nn.Parameter(torch.empty(in_features, r, dtype=torch.float32))
        self.lora_B = torch.nn.Parameter(torch.empty(r, out_features, dtype=torch.float32))
        
        # Initialize
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        base_out = self.base_linear(x)
        dtype = x.dtype
        # Cast parameters to match input dtype during forward pass
        lora_A_cast = self.lora_A.to(dtype)
        lora_B_cast = self.lora_B.to(dtype)
        lora_out = torch.matmul(torch.matmul(x, lora_A_cast), lora_B_cast) * self.scaling
        return base_out + lora_out

def train_lora_online(model, processor, trajectory, num_epochs=3, lr=1e-4):
    """Runs online gradient descent updates on the LoRA parameters using the trajectory."""
def train_lora_online(model, processor, trajectory: list[dict], num_epochs=3, lr=1e-4):
    """Executes online SFT on LoRA parameters using purely textual trajectory to eliminate vision model overhead."""
    # 防禦性檢查：12B 模型只做推理，不進行任何反向傳播 (SFT / TTT) 以防 OOM
    model_name = getattr(model.config, "_name_or_path", "").lower()
    if "12b" in model_name:
        print(f"[SFT Skip] Model '{model_name}' is a 12B model. 12B models only do inference. Skipping online SFT.")
        model.eval()
        return

    # 限制 online SFT 的軌跡長度，防止歷史步驟過長導致 backward pass 顯存吃緊 (OOM)
    # 一般化優化：只保留最近 6 步有效軌跡，確保顯存穩定
    if len(trajectory) > 6:
        print(f"[Online SFT] Trajectory too long ({len(trajectory)} steps). Truncating to last 6 steps to prevent OOM.")
        trajectory = trajectory[-6:]

    model.train()
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n]
    if not lora_params:
        print("[Online LoRA] No LoRA parameters found! Skipping training.")
        model.eval()
        return
        
    optimizer = torch.optim.AdamW(lora_params, lr=lr)
    
    print(f"[Online LoRA] Starting online SFT on {len(trajectory)} successful steps...")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        random.shuffle(trajectory)
        
        for step_data in trajectory:
            prompt = step_data["prompt"]
            response = step_data["response"]
            
            # Prepare inputs (pure text)
            # 1. Compute prompt length to mask labels
            prompt_inputs = processor(text=prompt, return_tensors="pt")
            prompt_len = prompt_inputs["input_ids"].shape[1]
            
            # 2. Process full sequence (prompt + response)
            full_text = prompt + response
            inputs = processor(text=full_text, return_tensors="pt")
            inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            labels = inputs["input_ids"].clone()
            labels[:, :prompt_len] = -100 # Mask prompt tokens
            inputs["labels"] = labels
            
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"[Online LoRA] Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss/len(trajectory):.4f}")
        
    model.eval()
    print("[Online LoRA] Online SFT complete! LoRA weights updated.")

def get_gemma4_model_and_processor(model_id: str, device="cuda"):
    """Loads Gemma 4 model and processor, wrapping attention projections in LoRALinear."""
    cache_key_model = f"model_{model_id}"
    cache_key_processor = f"processor_{model_id}"
    
    if cache_key_model in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key_model], _MODEL_CACHE[cache_key_processor]

    print(f"[Gemma4Agent] Loading processor and model from {model_id}...")

    from transformers import Gemma4Processor, Gemma4ForConditionalGeneration
    processor = Gemma4Processor.from_pretrained(model_id)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        low_cpu_mem_usage=True
    )
    
    # Freeze all parameters of the base model
    for param in model.parameters():
        param.requires_grad = False
        
    # Wrap attention layers in-place with LoRALinear (task03 style: q + v only, r=4)
    print(f"[Gemma4Agent] Injecting LoRALinear into q_proj + v_proj for {model_id}...")
    for layer in model.model.language_model.layers:
        if hasattr(layer.self_attn, "q_proj") and layer.self_attn.q_proj is not None:
            layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=4, alpha=8)
        if hasattr(layer.self_attn, "v_proj") and layer.self_attn.v_proj is not None:
            layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=4, alpha=8)
            
    # Set requires_grad=True explicitly for all LoRA parameters
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
            
    # Move model to device
    model = model.to(device)
    
    # Cache initial LoRA states so we can reset them for new games
    lora_state_dict = {k: v.clone() for k, v in model.state_dict().items() if "lora_" in k}
    _MODEL_CACHE[f"initial_lora_state_{model_id}"] = lora_state_dict

    _MODEL_CACHE[cache_key_model] = model
    _MODEL_CACHE[cache_key_processor] = processor
    print(f"[Gemma4Agent] Model {model_id} loaded and wrapped successfully on device: {model.device}!")
    return model, processor


# ==========================================
# 3. Robust Action Parser
# ==========================================
def parse_action_response(response_text: str, available_actions: Optional[list[Any]] = None) -> GameAction:
    """Parses model text response into a GameAction, with fallback logic for regex and random actions."""
    response_text = response_text.strip()
    
    # 1. Try parsing JSON block
    try:
        # Extract everything inside the first { ... }
        json_match = re.search(r"(\{.*\})", response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
            action_name = str(data.get("action", "")).upper().strip()
            reasoning = data.get("reasoning", "")
            
            # Check for JSON template bias: if model output action_name is biased/templated (often ACTION1),
            # but reasoning text explicitly details intent for a different action (e.g. ACTION2-7 or RESET).
            occurrences = []
            for act_name in ["RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"]:
                idx = reasoning.upper().rfind(act_name)
                if idx != -1:
                    occurrences.append((idx, act_name))
            
            inferred_action = None
            if occurrences:
                occurrences.sort(reverse=True)
                inferred_action = occurrences[0][1]
                
            # Special handling for ACTION6 click intent
            # ONLY override if action_name is NOT ACTION6 or x/y is missing/invalid
            has_valid_json_coords = (
                action_name == "ACTION6" 
                and "x" in data and "y" in data 
                and isinstance(data["x"], (int, float)) 
                and isinstance(data["y"], (int, float))
                and 0 <= int(data["x"]) <= 63 
                and 0 <= int(data["y"]) <= 63
            )
            
            if not has_valid_json_coords:
                if inferred_action == "ACTION6" or "COORDINATE" in reasoning.upper() or "CLICK" in reasoning.upper():
                    coords = re.findall(r"\(?\s*(\d+)\s*,\s*(\d+)\s*\)?", reasoning)
                    x, y = None, None
                    if coords:
                        x, y = int(coords[0][0]), int(coords[0][1])
                    else:
                        x_matches = re.findall(r"x\s*[:=]?\s*(\d+)", reasoning, re.IGNORECASE)
                        y_matches = re.findall(r"y\s*[:=]?\s*(\d+)", reasoning, re.IGNORECASE)
                        if x_matches and y_matches:
                            x, y = int(x_matches[0]), int(y_matches[0])
                    if x is not None and y is not None and max(x, y) <= 63:
                        print(f"[Parser Self-Correction] Overriding biased JSON action '{action_name}' to ACTION6 with coordinates x={x}, y={y} based on reasoning: '{reasoning}'")
                        action_name = "ACTION6"
                        data["x"] = x
                        data["y"] = y
                elif inferred_action and inferred_action != action_name:
                    print(f"[Parser Self-Correction] Overriding biased JSON action '{action_name}' to '{inferred_action}' based on reasoning: '{reasoning}'")
                    action_name = inferred_action

            action = None
            if action_name == "RESET":
                action = GameAction.RESET
            elif action_name == "ACTION1":
                action = GameAction.ACTION1
            elif action_name == "ACTION2":
                action = GameAction.ACTION2
            elif action_name == "ACTION3":
                action = GameAction.ACTION3
            elif action_name == "ACTION4":
                action = GameAction.ACTION4
            elif action_name == "ACTION5":
                action = GameAction.ACTION5
            elif action_name == "ACTION7":
                action = GameAction.ACTION7
            elif action_name == "ACTION6":
                action = GameAction.ACTION6
                x = int(data.get("x", 0))
                y = int(data.get("y", 0))
                x = max(0, min(63, x))
                y = max(0, min(63, y))
                action.set_data({"x": x, "y": y})

            if action is not None:
                action.reasoning = data.get("reasoning", "Parsed from JSON")
                action.pdca_plan = data.get("pdca_plan", "")
                action.pdca_check = data.get("pdca_check", "")
                action.pdca_act = data.get("pdca_act", "")
                return action
    except Exception:
        pass

    # 2. Regex fallback: Look for ACTION6 first
    if "ACTION6" in response_text:
        # Find coordinates pattern e.g., (x, y) or x=..., y=...
        coords = re.findall(r"\(?\s*(\d+)\s*,\s*(\d+)\s*\)?", response_text)
        if coords:
            x, y = int(coords[0][0]), int(coords[0][1])
        else:
            x_matches = re.findall(r"x\s*[:=]?\s*(\d+)", response_text, re.IGNORECASE)
            y_matches = re.findall(r"y\s*[:=]?\s*(\d+)", response_text, re.IGNORECASE)
            x = int(x_matches[0]) if x_matches else 0
            y = int(y_matches[0]) if y_matches else 0
        x = max(0, min(63, x))
        y = max(0, min(63, y))
        action = GameAction.ACTION6
        action.set_data({"x": x, "y": y})
        action.reasoning = f"Parsed ACTION6 at ({x}, {y}) via regex"
        return action

    # 3. Look for other action names in text
    for act_name in ["RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION7"]:
        if act_name in response_text:
            action = getattr(GameAction, act_name)
            action.reasoning = f"Parsed {act_name} via regex"
            return action

    # 4. Fallback to a valid action from available actions
    action_list = [
        GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
        GameAction.ACTION4, GameAction.ACTION5
    ]
    if available_actions:
        action_choice = random.choice(list(available_actions))
        action_id = action_choice.value if hasattr(action_choice, 'value') else int(action_choice)
        if action_id == 6:
            action = GameAction.ACTION6
            x, y = random.randint(0, 63), random.randint(0, 63)
            action.set_data({"x": x, "y": y})
            action.reasoning = "Fallback random ACTION6"
        else:
            action = GameAction.from_id(action_id)
            action.reasoning = f"Fallback random valid action: ACTION{action_id}"
        return action

    # 5. Ultimate fallback
    action = GameAction.ACTION1
    action.reasoning = "Fallback default ACTION1 due to parsing failure"
    return action


# ==========================================
# 4. MyAgent Class Implementation
# ==========================================
class MyAgent(Agent):
    """Gemma 4 Multimodal agent with visual and text hybrid representation and online ICL history."""

    MAX_ACTIONS = float('inf')
    _MAX_FRAMES = 10

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed % (2**32 - 1))
        self.start_time = time.time()

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Gemma 4 Agent initialized for game: {self.game_id}")

        self.current_score = -1
        self.action_history = []  # List of steps: {"action_desc": str, "reasoning": str, "prev_grid": np.ndarray, "prev_score": int, "outcome": str}
        self.action_queue = deque()  # Queue for multi-step action execution (BFS/State-Tree search results)
        self.probed_coordinates = set()  # Set of coordinates clicked historically in this level to compute entropy
        self.control_colors = set()  # Persistent control colors for the level (1 and 9 are controls in this game)
        self.movable_color = None  # Persistent movable shape color for the level
        self.target_color = None  # Persistent target marker color for the level

        # Graph-based 3-stage explorer & solver state
        self.graph = {}  # state_hash -> StateNode dict
        self.level_initial_state_hash = None
        self.last_state_hash = None
        self.last_action_taken = None
        self.solved_levels_paths = {}  # level_initial_state_hash -> list of action tuples

        # Lazy load model and processor globally (default to E2B)
        self.model, self.processor = get_gemma4_model_and_processor("google/gemma-4-E2B-it", self.device)

        # Reset LoRA weights if a new game started
        prev_game_id = _MODEL_CACHE.get("current_game_id")
        if prev_game_id != self.game_id:
            initial_state_key = "initial_lora_state_google/gemma-4-E2B-it"
            if initial_state_key in _MODEL_CACHE:
                print(f"[LoRA Reset] Game changed from {prev_game_id} to {self.game_id}. Resetting LoRA weights to initial state.")
                self.model.load_state_dict(_MODEL_CACHE[initial_state_key], strict=False)
            _MODEL_CACHE["current_game_id"] = self.game_id

    def append_frame(self, frame: FrameData) -> None:
        self.frames.append(frame)
        if len(self.frames) > self._MAX_FRAMES:
            self.frames = self.frames[-self._MAX_FRAMES:]
        if frame.guid:
            self.guid = frame.guid
        if hasattr(self, "recorder") and not self.is_playback:
            import json
            self.recorder.record(json.loads(frame.model_dump_json()))

    def _get_score(self, frame):
        return getattr(frame, 'score', None) or frame.levels_completed

    def _update_history_outcome(self, latest_frame):
        """Deduces outcome (Grid changed with spatial transformation details, score increased, or no change) of the last action taken."""
        if not self.action_history:
            return

        last_step = self.action_history[-1]
        if last_step["outcome"] is not None:
            return

        current_grid = np.array(latest_frame.frame[-1])
        prev_grid = last_step["prev_grid"]

        current_score = self._get_score(latest_frame)
        prev_score = last_step["prev_score"]

        # Parse action info for simplified logging
        action_desc = last_step["action_desc"]

        # Align ACTION6 to the nearest button/control coordinate to fix causal attribution bugs
        if "ACTION6" in action_desc:
            try:
                coords_str = action_desc.split("(")[1].split(")")[0]
                raw_x = int(coords_str.split("x=")[1].split(",")[0])
                raw_y = int(coords_str.split("y=")[1])
                
                h, w = prev_grid.shape
                scale_x = int(64 / w)
                scale_y = int(64 / h)
                scale = min(scale_x, scale_y)
                x_padding = int((64 - (w * scale)) / 2)
                y_padding = int((64 - (h * scale)) / 2)
                
                target_grid_x = int((raw_x - x_padding) / scale) if raw_x - x_padding >= 0 else -1
                target_grid_y = int((raw_y - y_padding) / scale) if raw_y - y_padding >= 0 else -1
                
                btn_coords = []
                for r in range(h):
                    for c in range(w):
                        if int(prev_grid[r, c]) in [1, 9]:
                            btn_coords.append((c, r))
                            
                if btn_coords:
                    best_btn = None
                    min_dist = float('inf')
                    for bc_x, bc_y in btn_coords:
                        dist = (bc_x - target_grid_x)**2 + (bc_y - target_grid_y)**2
                        if dist < min_dist:
                            min_dist = dist
                            best_btn = (bc_x, bc_y)
                    
                    if best_btn and min_dist <= 4:
                        btn_gx, btn_gy = best_btn
                        # Map back to display coordinates
                        display_x = x_padding + int((btn_gx + 0.5) * scale)
                        display_y = y_padding + int((btn_gy + 0.5) * scale)
                        
                        # Correct the description to the actual aligned coordinates executed
                        old_desc = action_desc
                        action_desc = f"ACTION6(x={display_x}, y={display_y})"
                        last_step["action_desc"] = action_desc
                        print(f"[Causal Alignment] Re-aligned historical log '{old_desc}' -> '{action_desc}' (grid: {btn_gx}, {btn_gy})")
            except Exception as e:
                print(f"[Causal Alignment] Error during alignment: {e}")

        if current_score > prev_score:
            outcome = f"[SUCCESS] Score increased to {current_score}."
            conclusion = "GOAL ALIGNMENT PROGRESS!"
        elif not np.array_equal(prev_grid[1:, :], current_grid[1:, :]):
            diff_indices = np.where(prev_grid[1:, :] != current_grid[1:, :])
            diff_rows = diff_indices[0] + 1
            diff_cols = diff_indices[1]
            changed_cells = len(diff_rows)
            unique_prev = np.unique(prev_grid[diff_rows, diff_cols])
            unique_curr = np.unique(current_grid[diff_rows, diff_cols])
            
            # Simple heuristic: if changes only happen at or adjacent to the clicked coordinate, it's local.
            # Otherwise, it's global transformation.
            is_local = True
            if "x=" in action_desc and "y=" in action_desc:
                try:
                    # extract x and y
                    coords_str = action_desc.split("(")[1].split(")")[0]
                    cx = int(coords_str.split("x=")[1].split(",")[0])
                    cy = int(coords_str.split("y=")[1])
                    for y_idx, x_idx in zip(diff_rows, diff_cols):
                        if abs(x_idx - cx) > 1 or abs(y_idx - cy) > 1:
                            is_local = False
                            break
                except Exception:
                    is_local = False
            else:
                is_local = False

            if is_local:
                outcome = f"[Grid changed locally] {changed_cells} cells modified near click. Colors: {list(unique_prev)} -> {list(unique_curr)}."
                conclusion = "Local button toggle only (No global movement/rotation)."
            else:
                outcome = f"[Grid changed globally] {changed_cells} cells modified. Colors: {list(unique_prev)} -> {list(unique_curr)}."
                conclusion = "ACTIVE GLOBAL CONTROL! Shape moved, rotated, or scaled."
                # PERSISTENT: Record control color and movable color!
                if "x=" in action_desc and "y=" in action_desc:
                    try:
                        coords_str = action_desc.split("(")[1].split(")")[0]
                        cx = int(coords_str.split("x=")[1].split(",")[0])
                        cy = int(coords_str.split("y=")[1])
                        c_color = int(prev_grid[cy, cx])
                        unique_colors, counts = np.unique(prev_grid, return_counts=True)
                        bg_color = int(unique_colors[np.argmax(counts)])
                        if c_color != bg_color:
                            self.control_colors.add(c_color)
                            print(f"[Persistent Colors] Added control color: {c_color}")
                    except Exception as e:
                        print(f"[Persistent Colors] Failed to extract control color: {e}")
                
                unique_colors, counts = np.unique(current_grid, return_counts=True)
                bg_color = int(unique_colors[np.argmax(counts)])
                changed_active = [c for c in (set(unique_prev.tolist()) | set(unique_curr.tolist())) if c != bg_color]
                if changed_active:
                    sorted_by_size = sorted(changed_active, key=lambda c: np.sum(current_grid == c), reverse=False)
                    potential_movable = [c for c in sorted_by_size if c not in self.control_colors and c not in [0, 5]]
                    if potential_movable and self.movable_color is None:
                        self.movable_color = potential_movable[0]
                        print(f"[Persistent Colors] Identified movable color: {self.movable_color}")
        else:
            outcome = "No change."
            conclusion = "Inactive coordinate / Static background or marker."

        last_step["outcome"] = outcome
        last_step["conclusion"] = conclusion

    def _get_history_text(self) -> str:
        """Constructs a compact cause-and-effect log of the explored actions (Occam's Razor)."""
        if not self.action_history:
            return "No actions taken yet in this level."

        lines = ["=== EXPLORED CAUSES & EFFECTS (物理嘗試結果總結) ==="]
        for i, step in enumerate(self.action_history, 1):
            h_line = f" - Step {i:02d}: Action: {step['action_desc']} -> Reaction: {step['outcome']} (Conclusion: {step.get('conclusion', 'Pending')})"
            lines.append(h_line)
        return "\n".join(lines)

    def _has_time_elapsed(self) -> bool:
        # Cap execution to 8 hours
        return (time.time() - self.start_time) >= 8 * 3600 - 5 * 60

    def is_done(self, frames, latest_frame) -> bool:
        return latest_frame.state is GameState.WIN or self._has_time_elapsed()

    def perform_bfs_sequence_search(self, grid, candidate_coords: list[tuple[int, int]], available_actions, failed_actions: set[str]) -> list[tuple[int, int]]:
        """Performs a local general geometric BFS to find a sequence of actions (up to depth 3) 
        that maximizes the overlap or alignment of movable objects to static targets.
        """
        import numpy as np
        import math
        from collections import deque
        
        # 1. Identify Background and active grid sizes
        grid = np.array(grid)
        H, W = grid.shape
        unique_colors, counts = np.unique(grid, return_counts=True)
        bg_color = int(unique_colors[np.argmax(counts)])
        
        # Find sorted active colors
        color_freq = dict(zip(unique_colors.tolist(), counts.tolist()))
        if bg_color in color_freq:
            del color_freq[bg_color]
        if not color_freq:
            return []
        sorted_colors = sorted(color_freq.keys(), key=lambda c: color_freq[c])

        # 2. Extract Connected Components ignoring color (spatial connectivity only)
        visited = np.zeros((H, W), dtype=bool)
        objects = []
        dy = [-1, -1, -1,  0, 0,  1, 1, 1]
        dx = [-1,  0,  1, -1, 1, -1, 0, 1]

        for y in range(H):
            for x in range(W):
                val = int(grid[y, x])
                if val != bg_color and not visited[y, x]:
                    component = []
                    queue = [(y, x)]
                    visited[y, x] = True
                    head = 0
                    while head < len(queue):
                        cy, cx = queue[head]
                        head += 1
                        component.append((cy, cx))
                        for i in range(8):
                            ny, nx = cy + dy[i], cx + dx[i]
                            if 0 <= ny < H and 0 <= nx < W:
                                if int(grid[ny, nx]) != bg_color and not visited[ny, nx]:
                                    visited[ny, nx] = True
                                    queue.append((ny, nx))
                    
                    ys = [pt[0] for pt in component]
                    xs = [pt[1] for pt in component]
                    min_y, max_y = min(ys), max(ys)
                    min_x, max_x = min(xs), max(xs)
                    w_size = max_x - min_x + 1
                    h_size = max_y - min_y + 1
                    
                    # Categorize shape structure based on bounding box
                    if len(component) == 1:
                        shape_type = "Single Cell"
                    elif h_size <= 2 and w_size <= 2:
                        shape_type = "Square Cluster / Dial Dial"
                    elif h_size == 1 and w_size > 1:
                        shape_type = "Horizontal Linear Slider/Line"
                    elif w_size == 1 and h_size > 1:
                        shape_type = "Vertical Linear Slider/Line"
                    elif w_size > h_size:
                        shape_type = "Horizontal Linear Slider/Line"
                    elif h_size > w_size:
                        shape_type = "Vertical Linear Slider/Line"
                    else:
                        shape_type = "Square Cluster / Dial Dial"

                    objects.append({
                        "colors": set(int(grid[py, px]) for py, px in component),
                        "pts": set(component),
                        "bbox": (min_y, max_y, min_x, max_x),
                        "w": w_size,
                        "h": h_size,
                        "shape": shape_type
                    })

        # Identify control colors from history click coordinates that actually caused global changes
        if self.control_colors:
            control_colors = self.control_colors
        else:
            control_colors = set()
            for step in self.action_history:
                desc = step["action_desc"]
                outcome = step.get("outcome")
                if "ACTION6" in desc and outcome and "changed globally" in outcome:
                    try:
                        coords_str = desc.split("(")[1].split(")")[0]
                        cx = int(coords_str.split("x=")[1].split(",")[0])
                        cy = int(coords_str.split("y=")[1])
                        # Find which object contains this coordinate
                        for obj in objects:
                            if (cy, cx) in obj["pts"]:
                                control_colors.update(obj["colors"])
                    except Exception:
                        pass

        # Determine movable color
        if self.movable_color is not None:
            movable_color = self.movable_color
        else:
            movable_color = None
            for i in range(len(self.action_history) - 1):
                step = self.action_history[i]
                next_step = self.action_history[i+1]
                if step.get("outcome") and "changed globally" in step["outcome"]:
                    p_grid = step["prev_grid"]
                    n_grid = next_step["prev_grid"]
                    diff = np.where(p_grid != n_grid)
                    if len(diff[0]) > 0:
                        changed_active = [c for c in (set(p_grid[diff].tolist()) | set(n_grid[diff].tolist())) if c != bg_color]
                        if changed_active:
                            sorted_by_size = sorted(changed_active, key=lambda c: np.sum(grid == c), reverse=True)
                            movable_color = sorted_by_size[0]
                            break

            if movable_color is None:
                # Fallback to rarest/second rarest logic
                if len(sorted_colors) > 1:
                    movable_color = sorted_colors[1]
                elif sorted_colors:
                    movable_color = sorted_colors[0]

        # Identify target color and target points
        if self.target_color is not None and self.target_color != movable_color:
            target_color = self.target_color
        else:
            possible_target_colors = [c for c in sorted_colors if c not in control_colors and c != movable_color]
            target_color = possible_target_colors[0] if possible_target_colors else sorted_colors[0]
        
        target_pts = set(zip(*np.where(grid == target_color)))
        movable_pts = set(zip(*np.where(grid == movable_color)))
        
        if not target_pts or not movable_pts:
            return []

        # Dominant color helper
        def get_dominant_color(obj):
            colors_in_obj = [int(grid[py, px]) for py, px in obj["pts"] if int(grid[py, px]) != bg_color]
            if not colors_in_obj:
                return None
            unique, counts = np.unique(colors_in_obj, return_counts=True)
            return unique[np.argmax(counts)]

        # 3. Simulate click reaction
        def simulate_click(current_grid, cx, cy):
            new_grid = current_grid.copy()
            m_ys, m_xs = np.where(new_grid == movable_color)
            if len(m_xs) == 0:
                return new_grid

            # Find closest object to click (cx, cy)
            clicked_obj = None
            min_dist = float('inf')
            for obj in objects:
                for py, px in obj["pts"]:
                    d = abs(py - cy) + abs(px - cx)
                    if d < min_dist:
                        min_dist = d
                        clicked_obj = obj
            
            if clicked_obj is None or min_dist > 3:
                return new_grid

            # Verify if it is indeed a control component (dominant color cannot be target or movable)
            dom_color = get_dominant_color(clicked_obj)
            is_control = (dom_color is not None and dom_color != target_color and dom_color != movable_color)
            if not is_control:
                return new_grid

            # Get movable center
            min_y, max_y = np.min(m_ys), np.max(m_ys)
            min_x, max_x = np.min(m_xs), np.max(m_xs)
            center_y, center_x = int((min_y + max_y) / 2), int((min_x + max_x) / 2)

            # Determine layout shape and perform shift/rotate simulation
            w_size, h_size = clicked_obj["w"], clicked_obj["h"]
            shape_type = clicked_obj["shape"]

            # Rotation: Square/dial/single cell
            if shape_type in ["Single Cell", "Square Cluster / Dial Dial"] or (w_size <= 2 and h_size <= 2):
                rotated_coords = []
                for py, px in zip(m_ys, m_xs):
                    dy, dx = py - center_y, px - center_x
                    ny, nx = center_y + dx, center_x - dy
                    if 0 <= ny < H and 0 <= nx < W:
                        rotated_coords.append((ny, nx))
                if rotated_coords:
                    new_grid[new_grid == movable_color] = bg_color
                    for ry, rx in rotated_coords:
                        new_grid[ry, rx] = movable_color

            # Slider translation/stretch simulation
            elif shape_type == "Horizontal Linear Slider/Line" or w_size > h_size:
                obj_center_x = (clicked_obj["bbox"][2] + clicked_obj["bbox"][3]) / 2
                shift_x = 3 if cx > obj_center_x else -3
                shifted_coords = []
                for py, px in zip(m_ys, m_xs):
                    nx = px + shift_x
                    if 0 <= nx < W:
                        shifted_coords.append((py, nx))
                if shifted_coords:
                    new_grid[new_grid == movable_color] = bg_color
                    for sy, sx in shifted_coords:
                        new_grid[sy, sx] = movable_color

            elif shape_type == "Vertical Linear Slider/Line" or h_size > w_size:
                obj_center_y = (clicked_obj["bbox"][0] + clicked_obj["bbox"][1]) / 2
                shift_y = 3 if cy > obj_center_y else -3
                shifted_coords = []
                for py, px in zip(m_ys, m_xs):
                    ny = py + shift_y
                    if 0 <= ny < H:
                        shifted_coords.append((ny, px))
                if shifted_coords:
                    new_grid[new_grid == movable_color] = bg_color
                    for sy, sx in shifted_coords:
                        new_grid[sy, sx] = movable_color

            return new_grid

        # 4. Alignment Metric
        def evaluate_alignment(current_grid):
            curr_movable = set(zip(*np.where(current_grid == movable_color)))
            if not curr_movable:
                return float('inf')
            overlap = len(curr_movable.intersection(target_pts))
            
            m_ys, m_xs = np.where(current_grid == movable_color)
            t_ys, t_xs = np.where(current_grid == target_color)
            if len(m_ys) == 0 or len(t_ys) == 0:
                return float('inf')
            center_m = (np.mean(m_ys), np.mean(m_xs))
            center_t = (np.mean(t_ys), np.mean(t_xs))
            dist = math.sqrt((center_m[0] - center_t[0])**2 + (center_m[1] - center_t[1])**2)
            
            return -overlap * 100 + dist

        # Automatically detect control coordinates to append to candidate_coords
        auto_candidates = []
        for obj in objects:
            # A component is a control if its dominant color is neither target nor movable
            dom_color = get_dominant_color(obj)
            if dom_color is not None and dom_color != target_color and dom_color != movable_color:
                min_y, max_y, min_x, max_x = obj["bbox"]
                center_y = int((min_y + max_y) / 2)
                center_x = int((min_x + max_x) / 2)
                
                if obj["shape"] == "Horizontal Linear Slider/Line":
                    left_pt = (center_x - 3, center_y)
                    right_pt = (center_x + 3, center_y)
                    left_pt = (max(min_x, min(max_x, left_pt[0])), center_y)
                    right_pt = (max(min_x, min(max_x, right_pt[0])), center_y)
                    auto_candidates.extend([left_pt, right_pt])
                elif obj["shape"] == "Vertical Linear Slider/Line":
                    top_pt = (center_x, center_y - 3)
                    bottom_pt = (center_x, center_y + 3)
                    top_pt = (center_x, max(min_y, min(max_y, top_pt[1])))
                    bottom_pt = (center_x, max(min_y, min(max_y, bottom_pt[1])))
                    auto_candidates.extend([top_pt, bottom_pt])
                else:
                    auto_candidates.append((center_x, center_y))

        # Merge and deduplicate candidates, filtering out failed coordinates
        all_candidates = []
        seen = set()
        for cx, cy in list(candidate_coords) + auto_candidates:
            if 0 <= cx < W and 0 <= cy < H:
                if (cx, cy) not in seen:
                    action_desc = f"ACTION6(x={cx}, y={cy})"
                    if action_desc not in failed_actions:
                        seen.add((cx, cy))
                        all_candidates.append((cx, cy))
        
        if not all_candidates:
            for y in range(1, H, 3):
                for x in range(1, W, 3):
                    if int(grid[y, x]) != bg_color:
                        all_candidates.append((x, y))

        # 5. BFS state tree search
        best_seq = []
        best_score = evaluate_alignment(grid)
        
        queue = deque([(grid.copy(), [])])
        visited_states = {grid.tobytes()}
        
        max_depth = 3
        while queue:
            curr_state, path = queue.popleft()
            if len(path) >= max_depth:
                continue
                
            for cx, cy in all_candidates:
                next_state = simulate_click(curr_state, cx, cy)
                state_bytes = next_state.tobytes()
                
                if state_bytes not in visited_states:
                    visited_states.add(state_bytes)
                    next_path = path + [(cx, cy)]
                    score = evaluate_alignment(next_state)
                    
                    if score < best_score:
                        best_score = score
                        best_seq = next_path
                        
                    queue.append((next_state, next_path))
                    
        return best_seq

    def choose_rareness_exploration_action(self, grid, failed_actions, available_actions) -> Optional[GameAction]:
        import numpy as np
        import random
        from arcengine import GameAction
        
        is_action6_valid = True
        if available_actions is not None:
            valid_ids = [a.value if hasattr(a, 'value') else int(a) for a in available_actions]
            if 6 not in valid_ids:
                is_action6_valid = False
                
        if is_action6_valid:
            grid = np.array(grid)
            unique_colors, counts = np.unique(grid, return_counts=True)
            color_counts = dict(zip(unique_colors, counts))
            
            sorted_colors = sorted(color_counts.keys(), key=lambda c: color_counts[c])
            max_color = max(color_counts, key=color_counts.get)
            
            for color in sorted_colors:
                if color == max_color:
                    continue
                    
                ys, xs = np.where(grid == color)
                coords_for_color = list(zip(xs, ys))
                random.shuffle(coords_for_color)
                
                for x, y in coords_for_color:
                    action_desc = f"ACTION6(x={x}, y={y})"
                    if action_desc not in failed_actions:
                        action = GameAction.ACTION6
                        action.set_data({"x": int(x), "y": int(y)})
                        action.reasoning = f"[Rareness-First Exploration] Click rarest color {color} at ({x}, {y})"
                        return action

        candidate_discrete = []
        for act_choice in [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION7]:
            act_id = act_choice.value if hasattr(act_choice, 'value') else int(act_choice)
            if available_actions and act_id not in [a.value if hasattr(a, 'value') else int(a) for a in available_actions]:
                continue
            if act_choice.name not in failed_actions:
                candidate_discrete.append(act_choice)
                
        if candidate_discrete:
            override_action = random.choice(candidate_discrete)
            override_action.reasoning = f"[Discrete Exploration] Try action {override_action.name}"
            return override_action
            
        return None

    def action_tuple_to_game_action(self, action_tuple) -> GameAction:
        action_type, detail = action_tuple
        if action_type == "DISCRETE":
            action = getattr(GameAction, detail)
            action.reasoning = f"Graph: {detail}"
            return action
        elif action_type == "CLICK":
            cx, cy = detail
            action = GameAction.ACTION6
            action.set_data({"x": cx, "y": cy})
            action.reasoning = f"Graph Click: ({cx}, {cy})"
            return action
        raise ValueError(f"Unknown action type: {action_type}")

    def game_action_to_action_tuple(self, game_action: GameAction) -> tuple[str, Any]:
        if game_action.name == "ACTION6" and hasattr(game_action, "action_data") and game_action.action_data:
            data = game_action.action_data.model_dump()
            return ("CLICK", (int(data.get("x", 0)), int(data.get("y", 0))))
        return ("DISCRETE", game_action.name)

    def action_tuple_to_desc(self, action_tuple) -> str:
        action_type, detail = action_tuple
        if action_type == "DISCRETE":
            return detail
        elif action_type == "CLICK":
            cx, cy = detail
            return f"ACTION6(x={cx}, y={cy})"
        return str(action_tuple)

    def detect_control_coordinates(self, grid) -> list[tuple[int, int]]:
        grid = np.array(grid)
        H, W = grid.shape
        unique_colors, counts = np.unique(grid, return_counts=True)
        bg_color = int(unique_colors[np.argmax(counts)])
        
        visited = np.zeros((H, W), dtype=bool)
        components = []
        dy = [-1, -1, -1,  0, 0,  1, 1, 1]
        dx = [-1,  0,  1, -1, 1, -1, 0, 1]
        
        for y in range(H):
            for x in range(W):
                val = int(grid[y, x])
                if val != bg_color and not visited[y, x]:
                    comp = []
                    q = [(y, x)]
                    visited[y, x] = True
                    head = 0
                    while head < len(q):
                        cy, cx = q[head]
                        head += 1
                        comp.append((cy, cx))
                        for i in range(8):
                            ny, nx = cy + dy[i], cx + dx[i]
                            if 0 <= ny < H and 0 <= nx < W:
                                if int(grid[ny, nx]) == val and not visited[ny, nx]:
                                    visited[ny, nx] = True
                                    q.append((ny, nx))
                    components.append((val, comp))
                    
        coords = []
        for val, comp in components:
            ys = [pt[0] for pt in comp]
            xs = [pt[1] for pt in comp]
            min_y, max_y = min(ys), max(ys)
            min_x, max_x = min(xs), max(xs)
            center_y = int(np.mean(ys))
            center_x = int(np.mean(xs))
            
            coords.append((center_x, center_y))
            h_size = max_y - min_y + 1
            w_size = max_x - min_x + 1
            if w_size >= 4:
                coords.append((max(min_x, center_x - 3), center_y))
                coords.append((min(max_x, center_x + 3), center_y))
            if h_size >= 4:
                coords.append((center_x, max(min_y, center_y - 3)))
                coords.append((center_x, min(max_y, center_y + 3)))
                
        # Dynamically define target, wall, and movable colors to avoid clicking them
        movable_c = getattr(self, "movable_color", None)
        target_c = getattr(self, "target_color", None)
        sorted_by_freq = sorted(range(len(unique_colors)), key=lambda idx: counts[idx], reverse=True)
        wall_c = 5  # fallback
        for idx in sorted_by_freq:
            c_val = int(unique_colors[idx])
            if c_val not in [bg_color, movable_c, target_c]:
                wall_c = c_val
                break

        for y in range(H):
            for x in range(W):
                val = int(grid[y, x])
                if val not in [bg_color, wall_c, target_c, movable_c]:
                    coords.append((x, y))
                    
        deduped = []
        seen = set()
        for cx, cy in coords:
            cx = max(0, min(W - 1, cx))
            cy = max(0, min(H - 1, cy))
            if (cx, cy) not in seen:
                seen.add((cx, cy))
                deduped.append((cx, cy))
                
        if not deduped:
            for y in range(0, H, 3):
                for x in range(0, W, 3):
                    if int(grid[y, x]) != bg_color:
                        deduped.append((x, y))
        if not deduped:
            deduped.append((W // 2, H // 2))
        return deduped

    def get_possible_actions(self, grid, available_actions) -> list[tuple[str, Any]]:
        actions = []
        if not available_actions:
            avail_ids = [1, 2, 3, 4, 5, 6, 7]
        else:
            avail_ids = [a.value if hasattr(a, 'value') else int(a) for a in available_actions]
        for aid in avail_ids:
            if aid in [1, 2, 3, 4, 5, 7]:
                actions.append(("DISCRETE", f"ACTION{aid}"))
            elif aid == 6:
                coords = self.detect_control_coordinates(grid)
                for cx, cy in coords:
                    actions.append(("CLICK", (cx, cy)))
        return actions

    def find_shortest_path(self, start_hash, end_hash) -> Optional[list[Any]]:
        if start_hash not in self.graph or end_hash not in self.graph:
            return None
        if start_hash == end_hash:
            return []
        queue = deque([(start_hash, [])])
        visited = {start_hash}
        while queue:
            node_hash, path = queue.popleft()
            node = self.graph[node_hash]
            for action, dest_hash in node["edges"].items():
                if dest_hash == end_hash:
                    return path + [action]
                if dest_hash not in visited:
                    visited.add(dest_hash)
                    queue.append((dest_hash, path + [action]))
        return None

    def find_shortest_path_to_frontier(self, start_hash) -> tuple[Optional[list[str]], list[tuple[str, Any]]]:
        if start_hash not in self.graph:
            return None, []
        queue = deque([[start_hash]])
        visited = {start_hash}
        while queue:
            path = queue.popleft()
            node_hash = path[-1]
            node = self.graph[node_hash]
            
            unexplored = [a for a in node["possible_actions"] if a not in node["edges"]]
            if unexplored:
                return path, unexplored
                
            for action, dest_hash in node["edges"].items():
                if dest_hash not in visited:
                    visited.add(dest_hash)
                    queue.append(path + [dest_hash])
        return None, []

    def choose_action(self, frames, latest_frame) -> GameAction:
        try:
            # 1. Update outcome of the previous action
            self._update_history_outcome(latest_frame)
            
            current_grid = np.array(latest_frame.frame[-1])
            current_hash = hashlib.sha256(current_grid.tobytes()).hexdigest()
            current_level = self._get_score(latest_frame)

            # Record last transition in the graph
            if self.last_state_hash is not None and self.last_action_taken is not None:
                if self.last_state_hash in self.graph:
                    self.graph[self.last_state_hash]["edges"][self.last_action_taken] = current_hash
                    desc = self.action_tuple_to_desc(self.last_action_taken)
                    print(f"[Graph Memory] Recorded transition: {self.last_state_hash[:8]} --({desc})--> {current_hash[:8]}")

            # 2. Check if level is updated or started
            if current_level != self.current_score:
                print(f"[{self.game_id}] Level updated or started: score changed from {self.current_score} to {current_level}")
                
                # Check if we successfully completed the previous level
                if self.current_score != -1 and current_level > self.current_score:
                    # Save winning path of previous level
                    if self.level_initial_state_hash is not None and self.last_state_hash is not None:
                        path_to_prev = self.find_shortest_path(self.level_initial_state_hash, self.last_state_hash)
                        if path_to_prev is not None:
                            winning_path = path_to_prev + [self.last_action_taken]
                            self.solved_levels_paths[self.level_initial_state_hash] = winning_path
                            print(f"[Graph Solver] Saved winning path of length {len(winning_path)} for level initial state {self.level_initial_state_hash}")

                    # Reset LoRA weights to initial state to prevent cross-level interference
                    initial_state_key = "initial_lora_state_google/gemma-4-E2B-it"
                    if initial_state_key in _MODEL_CACHE:
                        print(f"[LoRA Reset] Level upgraded. Resetting LoRA weights to initial state.")
                        self.model.load_state_dict(_MODEL_CACHE[initial_state_key], strict=False)

                # Initialize state variables for the new level
                self.graph = {}
                self.level_initial_state_hash = current_hash
                self.current_score = current_level
                self.action_queue.clear()
                self.action_history.clear()
                self.probed_coordinates.clear()
                self.control_colors = set()
                self.movable_color = None
                self.target_color = None

            # Ensure current state is in the graph
            if current_hash not in self.graph:
                self.graph[current_hash] = {
                    "state_hash": current_hash,
                    "grid": current_grid,
                    "score": current_level,
                    "possible_actions": self.get_possible_actions(current_grid, latest_frame.available_actions),
                    "edges": {}
                }

            # 3. Handle Game Reset
            if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                # If level failed, learn from successful steps online before reset
                if latest_frame.state == GameState.GAME_OVER and self.action_history:
                    positive_steps = [
                        step for step in self.action_history
                        if step.get("outcome") is not None and step["outcome"] != "No change."
                    ]
                    if positive_steps:
                        is_e2b = getattr(self.model, "config", None) and getattr(self.model.config, "_name_or_path", "").lower().find("12b") == -1
                        if is_e2b:
                            print(f"[{self.game_id}] Level failed. Running online SFT on {len(positive_steps)} positive steps...")
                            try:
                                train_lora_online(self.model, self.processor, positive_steps, num_epochs=3, lr=1e-4)
                            except Exception as e:
                                print(f"[Online SFT Error] SFT failed: {e}")
                                
                self.action_queue.clear()
                self.action_history.clear()
                self.probed_coordinates.clear()
                self.control_colors = set()
                self.movable_color = None
                self.target_color = None
                self.last_state_hash = None
                self.last_action_taken = None
                action = GameAction.RESET
                action.reasoning = "Game needs reset."
                return action

            # 4. Phase 3: Perfect Solving if winning path is known for this level
            if self.level_initial_state_hash in self.solved_levels_paths:
                winning_path = self.solved_levels_paths[self.level_initial_state_hash]
                if current_hash == self.level_initial_state_hash:
                    # We are at the start of the level, load and execute the entire path
                    self.action_queue.extend(winning_path)
                    print(f"[Graph Solver] Replaying winning path of length {len(winning_path)}: {[self.action_tuple_to_desc(a) for a in winning_path]}")
                else:
                    # We are in a different state, reset the level so we start from the initial state
                    print(f"[Graph Solver] Winning path known. Resetting to initial state to replay.")
                    self.last_state_hash = None
                    self.last_action_taken = None
                    action = GameAction.RESET
                    action.reasoning = "Resetting to execute winning path from level start."
                    return action

            # 5. Execute queued actions from navigation or planning
            if self.action_queue:
                action_tuple = self.action_queue.popleft()
                action = self.action_tuple_to_game_action(action_tuple)
                self.last_state_hash = current_hash
                self.last_action_taken = action_tuple
                
                # Record in action_history for SFT / logging
                step_record = {
                    "action_desc": self.action_tuple_to_desc(action_tuple),
                    "reasoning": action.reasoning,
                    "pdca_plan": "Graph navigation/solve replay",
                    "pdca_check": "Executing planned graph transition",
                    "pdca_act": "Keep moving",
                    "prev_grid": current_grid,
                    "prev_score": current_level,
                    "outcome": None,
                    "prompt": "",
                    "response": ""
                }
                self.action_history.append(step_record)
                if len(self.action_history) > 15:
                    self.action_history.pop(0)
                print(f"[{self.game_id}] Executing queued action: {step_record['action_desc']}")
                return action

            # 6. Phase 2: Frontier Exploration
            path_to_frontier, unexplored_actions = self.find_shortest_path_to_frontier(current_hash)
            
            if path_to_frontier is None or not unexplored_actions:
                # All reachable states/actions fully explored
                print(f"[{self.game_id}] State space exhausted. Resetting to search other paths.")
                self.last_state_hash = None
                self.last_action_taken = None
                action = GameAction.RESET
                action.reasoning = "Exploration exhausted, resetting."
                return action

            # If we need to navigate to reach the frontier node
            if len(path_to_frontier) > 1:
                # Reconstruct the sequence of actions to navigate to the frontier node
                nav_actions = []
                for i in range(len(path_to_frontier) - 1):
                    u_hash = path_to_frontier[i]
                    v_hash = path_to_frontier[i+1]
                    # Find action that leads from u to v
                    found_act = None
                    for act, dest in self.graph[u_hash]["edges"].items():
                        if dest == v_hash:
                            found_act = act
                            break
                    if found_act is not None:
                        nav_actions.append(found_act)
                
                self.action_queue.extend(nav_actions)
                print(f"[Graph Explorer] Navigating {len(nav_actions)} steps to frontier: {[self.action_tuple_to_desc(a) for a in nav_actions]}")
                
                # Pop and execute the first navigation step
                action_tuple = self.action_queue.popleft()
                action = self.action_tuple_to_game_action(action_tuple)
                self.last_state_hash = current_hash
                self.last_action_taken = action_tuple
                
                step_record = {
                    "action_desc": self.action_tuple_to_desc(action_tuple),
                    "reasoning": action.reasoning,
                    "pdca_plan": f"Navigate to frontier node {path_to_frontier[-1][:8]}",
                    "pdca_check": "Executing navigation",
                    "pdca_act": "Continue navigation",
                    "prev_grid": current_grid,
                    "prev_score": current_level,
                    "outcome": None,
                    "prompt": "",
                    "response": ""
                }
                self.action_history.append(step_record)
                if len(self.action_history) > 15:
                    self.action_history.pop(0)
                return action

            # We are currently at the frontier node. Choose one unexplored action.
            # Ask the LLM to get a heuristic recommendation!
            # Generate representation
            grid_text = grid_to_text_representation(current_grid)
            available_actions = getattr(latest_frame, 'available_actions', None)
            avail_ids = [a.value if hasattr(a, 'value') else int(a) for a in (available_actions or [])]
            avail_desc = ", ".join(f"ACTION{a}" if a != 6 else "ACTION6" for a in (available_actions or []))

            # Helper alignment variables
            H, W = current_grid.shape
            unique_colors, counts = np.unique(current_grid, return_counts=True)
            bg_color = int(unique_colors[np.argmax(counts)])
            
            color_freq = dict(zip(unique_colors.tolist(), counts.tolist()))
            if bg_color in color_freq:
                del color_freq[bg_color]
            sorted_colors = sorted(color_freq.keys(), key=lambda c: color_freq[c])

            # Adjacency-based target/movable shape identification for context injection
            visited_comp = np.zeros((H, W), dtype=bool)
            color_components = []
            dy_comp = [-1, -1, -1,  0, 0,  1, 1, 1]
            dx_comp = [-1,  0,  1, -1, 1, -1, 0, 1]

            for y_idx in range(H):
                for x_idx in range(W):
                    val = int(current_grid[y_idx, x_idx])
                    if val != bg_color and not visited_comp[y_idx, x_idx]:
                        comp_pts = []
                        queue_comp = [(y_idx, x_idx)]
                        visited_comp[y_idx, x_idx] = True
                        head_comp = 0
                        while head_comp < len(queue_comp):
                            cy, cx = queue_comp[head_comp]
                            head_comp += 1
                            comp_pts.append((cy, cx))
                            for i in range(8):
                                ny, nx = cy + dy_comp[i], cx + dx_comp[i]
                                if 0 <= ny < H and 0 <= nx < W:
                                    if int(current_grid[ny, nx]) == val and not visited_comp[ny, nx]:
                                        visited_comp[ny, nx] = True
                                        queue_comp.append((ny, nx))
                        color_components.append({
                            "color": val,
                            "pts": set(comp_pts)
                        })

            color_comps_map = {}
            for comp in color_components:
                c = comp["color"]
                if c == bg_color or c in self.control_colors:
                    continue
                color_comps_map.setdefault(c, []).append(comp)

            core_colors = [c for c in color_comps_map if c not in [0, 5, bg_color] and c not in self.control_colors]
            target_colors_set = set()
            movable_colors_set = set()

            for c in core_colors:
                comps = color_comps_map[c]
                has_touching = False
                has_isolated = False
                for comp in comps:
                    touches_other = False
                    for other_c in core_colors:
                        if other_c == c:
                            continue
                        for other_comp in color_comps_map[other_c]:
                            is_adj = False
                            for py, px in comp["pts"]:
                                for d_y in [-1, 0, 1]:
                                    for d_x in [-1, 0, 1]:
                                        if (py + d_y, px + d_x) in other_comp["pts"]:
                                            is_adj = True
                                            break
                                    if is_adj:
                                        break
                                if is_adj:
                                    break
                            if is_adj:
                                touches_other = True
                                break
                        if touches_other:
                            break
                    if touches_other:
                        has_touching = True
                    else:
                        has_isolated = True

                if has_touching and has_isolated:
                    target_colors_set.add(c)
                elif has_touching and not has_isolated:
                    movable_colors_set.add(c)
                elif not has_touching and has_isolated:
                    target_colors_set.add(c)

            if self.movable_color is None:
                if movable_colors_set:
                    self.movable_color = list(movable_colors_set)[0]
                else:
                    if len(sorted_colors) > 1:
                        self.movable_color = sorted_colors[1]
                    elif sorted_colors:
                        self.movable_color = sorted_colors[0]

            if self.target_color is None or self.target_color == self.movable_color:
                if target_colors_set:
                    self.target_color = list(target_colors_set)[0]
                else:
                    possible_target_colors = [c for c in sorted_colors if c not in self.control_colors and c != self.movable_color]
                    self.target_color = possible_target_colors[0] if possible_target_colors else sorted_colors[0]

            movable_color = self.movable_color
            target_color = self.target_color

            # Dynamically detect wall color
            wall_color = 5
            sorted_by_freq = sorted(range(len(unique_colors)), key=lambda idx: counts[idx], reverse=True)
            for idx in sorted_by_freq:
                c_val = int(unique_colors[idx])
                if c_val not in [bg_color, movable_color, target_color, 1, 9]:
                    wall_color = c_val
                    break

            # Calculate target gap
            ys_t, xs_t = np.where(current_grid == target_color)
            gap_text = "No target or movable shapes identified."
            if len(ys_t) > 0:
                tooth_pts = []
                target_pts = []
                for y, x in zip(ys_t, xs_t):
                    is_tooth = False
                    for dy_adj, dx_adj in [(-1,0), (1,0), (0,-1), (0,1)]:
                        ny, nx = y + dy_adj, x + dx_adj
                        if 0 <= ny < H and 0 <= nx < W:
                            if current_grid[ny, nx] == movable_color:
                                is_tooth = True
                    if is_tooth:
                        tooth_pts.append((y, x))
                    else:
                        target_pts.append((y, x))
                
                if tooth_pts and target_pts:
                    t_y, t_x = np.mean(tooth_pts, axis=0)
                    g_y, g_x = np.mean(target_pts, axis=0)
                    dx_gap = g_x - t_x
                    dy_gap = g_y - t_y
                else:
                    m_ys, m_xs = np.where(current_grid == movable_color)
                    if len(m_ys) > 0:
                        dx_gap = np.mean(xs_t) - np.mean(m_xs)
                        dy_gap = np.mean(ys_t) - np.mean(m_ys)
                    else:
                        dx_gap, dy_gap = 0.0, 0.0
                
                gap_text = (
                    f"- Translation Gap: dx={dx_gap:.1f} columns, dy={dy_gap:.1f} rows\n"
                    f"- Goal orientation alignment: Compare movable shape color {movable_color} and target marker color {target_color}."
                )

            # Build transition table logs from action history
            causal_lines = []
            seen_actions = set()
            for i in range(len(self.action_history)):
                step_data = self.action_history[i]
                action_str = step_data.get("action_desc") or ""
                if "ACTION6" not in action_str:
                    continue
                if action_str in seen_actions:
                    continue
                seen_actions.add(action_str)
                
                grid_prev = step_data.get("prev_grid")
                grid_post = self.action_history[i+1].get("prev_grid") if i + 1 < len(self.action_history) else current_grid
                    
                effect_desc = "No visible change."
                if grid_prev is not None and grid_post is not None:
                    try:
                        m_ys_prev, m_xs_prev = np.where(grid_prev == movable_color)
                        m_ys_post, m_xs_post = np.where(grid_post == movable_color)
                        if len(m_ys_prev) > 0 and len(m_ys_post) > 0:
                            dy_shift = np.mean(m_ys_post) - np.mean(m_ys_prev)
                            dx_shift = np.mean(m_xs_post) - np.mean(m_xs_prev)
                            if abs(dx_shift) > 0.1 or abs(dy_shift) > 0.1:
                                effect_desc = f"Translates movable shape by dx={dx_shift:.1f}, dy={dy_shift:.1f}"
                            else:
                                old_rel = sorted([(y - int(np.mean(m_ys_prev)), x - int(np.mean(m_xs_prev))) for y, x in zip(m_ys_prev, m_xs_prev)])
                                new_rel = sorted([(y - int(np.mean(m_ys_post)), x - int(np.mean(m_xs_post))) for y, x in zip(m_ys_post, m_xs_post)])
                                if old_rel != new_rel:
                                    effect_desc = "Rotates/reorients movable shape"
                                else:
                                    effect_desc = "No visible effect on movable shape"
                    except Exception as parse_err:
                        effect_desc = f"Error: {parse_err}"
                if step_data.get("outcome") == "No change.":
                    effect_desc = "No change."
                causal_lines.append(f"  - {action_str} -> Causal Effect: {effect_desc}")
            causal_table_str = "\n".join(causal_lines) if causal_lines else "  No historical controls tested in this level yet."

            # Active buttons detection
            detected_buttons = []
            h, w = current_grid.shape
            scale_x = int(64 / w)
            scale_y = int(64 / h)
            scale = min(scale_x, scale_y)
            x_padding = int((64 - (w * scale)) / 2)
            y_padding = int((64 - (h * scale)) / 2)
            
            visited_btn = np.zeros((h, w), dtype=bool)
            for r in range(h):
                for c in range(w):
                    val = int(current_grid[r, c])
                    if val not in [bg_color, wall_color, target_color, movable_color] and not visited_btn[r, c]:
                        comp = []
                        q = [(r, c)]
                        visited_btn[r, c] = True
                        head = 0
                        while head < len(q):
                            cy, cx = q[head]
                            head += 1
                            comp.append((cy, cx))
                            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                                nr, nc = cy + dr, cx + dc
                                if 0 <= nr < h and 0 <= nc < w:
                                    if int(current_grid[nr, nc]) == val and not visited_btn[nr, nc]:
                                        visited_btn[nr, nc] = True
                                        q.append((nr, nc))
                        ys = [pt[0] for pt in comp]
                        xs = [pt[1] for pt in comp]
                        center_gy = int(np.mean(ys))
                        center_gx = int(np.mean(xs))
                        disp_x = x_padding + int((center_gx + 0.5) * scale)
                        disp_y = y_padding + int((center_gy + 0.5) * scale)
                        detected_buttons.append({
                            "color": val,
                            "display_center": (disp_x, disp_y),
                            "bbox": (min(ys), max(ys), min(xs), max(xs))
                        })
                        
            buttons_text = ""
            if detected_buttons:
                buttons_text = "=== DETECTED ACTIVE BUTTONS/CONTROLS ===\n"
                for idx, btn in enumerate(detected_buttons, 1):
                    desc = "Button (Transfer Width/Height)" if btn["color"] == 9 else "Slide Key (Directional Control)"
                    buttons_text += (
                        f"Control {idx}: {desc}\n"
                        f"  - Color: {btn['color']}\n"
                        f"  - Grid bbox: rows {btn['bbox'][0]} to {btn['bbox'][1]}, cols {btn['bbox'][2]} to {btn['bbox'][3]}\n"
                        f"  - RECOMMENDED coordinate to click: ACTION6(x={btn['display_center'][0]}, y={btn['display_center'][1]})\n"
                    )
            else:
                buttons_text = "No active controls detected."

            history_text = self._get_history_text()
            user_content_text = (
                f"History of actions in this level:\n"
                f"{history_text}\n\n"
                f"=== ALIGNMENT TARGET GAP ===\n"
                f"{gap_text}\n\n"
                f"{buttons_text}\n\n"
                f"=== CAUSAL TRANSITION TABLE (Offline Simulation) ===\n"
                f"{causal_table_str}\n\n"
                f"Current Grid Text Map:\n"
                f"{grid_text}\n\n"
                f"Valid actions at this step: {avail_desc}"
            )

            # Determine entropy & model routing
            probed_count = 0
            for btn in detected_buttons:
                disp_x, disp_y = btn["display_center"]
                has_clicked = False
                for cx, cy in self.probed_coordinates:
                    if abs(cx - disp_x) <= scale and abs(cy - disp_y) <= scale:
                        has_clicked = True
                        break
                if has_clicked:
                    probed_count += 1
            total_controls = len(detected_buttons)
            entropy = 1.0 - (probed_count / total_controls) if total_controls > 0 else 0.0

            if entropy > 0.2:
                model_id = "google/gemma-4-E2B-it"
                phase_title = "=== SYSTEM EXPLORATION PHASE (掃盲探索階段) ==="
                phase_desc = (
                    f"- Current System Entropy: {entropy:.2f} (High Uncertainty)\n"
                    f"- Goal: Probe unexplored control coordinates to identify their causal effects and complete transition table."
                )
            else:
                model_id = "google/gemma-4-12b-it"
                phase_title = "=== PRECISION EXECUTION PLAN PHASE (精準對齊執行階段) ==="
                phase_desc = (
                    f"- Current System Entropy: {entropy:.2f} (Uncertainty Resolved)\n"
                    f"- Goal: The causal table is complete. Choose the correct action to align the shape with the targets."
                )

            active_model, active_processor = get_gemma4_model_and_processor(model_id, self.device)

            prev_game_id = _MODEL_CACHE.get("current_game_id")
            if prev_game_id != self.game_id:
                initial_state_key = f"initial_lora_state_{model_id}"
                if initial_state_key in _MODEL_CACHE:
                    active_model.load_state_dict(_MODEL_CACHE[initial_state_key], strict=False)
                _MODEL_CACHE["current_game_id"] = self.game_id

            user_content_text = f"{phase_title}\n{phase_desc}\n\n{user_content_text}"

            unexplored_desc = "\n".join(f"- {self.action_tuple_to_desc(act)}" for act in unexplored_actions)
            user_content_text += (
                f"\n\n=== EXPLORATION FRONTIER ===\n"
                f"The following actions are UNEXPLORED from the current state. "
                f"You MUST choose or recommend one of them to proceed:\n{unexplored_desc}"
            )
            user_content_text += "\n\nAnalyze the grid text map, follow the exploration rules, and return your chosen action as a JSON block."

            # Dynamically format example and restriction based on availability of ACTION6
            if 6 in avail_ids:
                format_example = (
                    "{\n"
                    '  "pdca_plan": "(繁體中文) 幾何朝向與尺寸對齊分析、滑軌/旋鈕佈局假設與本步測試計畫",\n'
                    '  "pdca_check": "(繁體中文) 檢查上一步動作是觸發局部變色還是全局幾何位移（旋轉/伸縮），評估與目標之對齊差距",\n'
                    '  "pdca_act": "(繁體中文) 根據全局幾何變化結果調整後的下一步控制方向",\n'
                    '  "action": "ACTION6",\n'
                    '  "x": 10,\n'
                    '  "y": 34,\n'
                    '  "reasoning": "Clicking the dial control to rotate the central shape to align with the target marker orientation.",\n'
                    '  "candidate_control_coords": [[10, 34], [10, 27], [2, 6]]\n'
                    "}\n"
                )
                action_restriction = ""
            else:
                format_example = (
                    "{\n"
                    '  "pdca_plan": "(繁體中文) 幾何朝向與尺寸對齊分析、操控游標移動與本步測試計畫",\n'
                    '  "pdca_check": "(繁體中文) 檢查上一步動作是觸發局部變色還是全局幾何位移（旋轉/伸縮），評估與目標之對齊差距",\n'
                    '  "pdca_act": "(繁體中文) 根據全局幾何變化結果調整後的下一步控制方向",\n'
                    '  "action": "ACTION2",\n'
                    '  "reasoning": "Moving the cursor towards the directional buttons to trigger a shape rotation.",\n'
                    '  "candidate_control_coords": []\n'
                    "}\n"
                )
                action_restriction = (
                    "\n=== CRITICAL OPERATION CONSTRAINT ===\n"
                    "ACTION6 (coordinate clicking) is NOT available in this level! "
                    "Do NOT output ACTION6 in your JSON. You must choose a discrete action (e.g. ACTION1, ACTION2, ACTION3, ACTION4) "
                    "to navigate the cursor to interact with the controls."
                )

            system_content = (
                "You are an elite scientific reasoning agent playing ARC-AGI-3 interactive puzzle games.\n"
                "You do NOT know the game rules in advance. Your goal is to DISCOVER the rules through \n"
                "systematic exploration and solve each level using a strict PDCA (Plan-Do-Check-Act) loop.\n\n"
                "=== COORDINATE SYSTEM ===\n"
                "The grid is H rows x W columns. Coordinates:\n"
                "  - x = column index (0 = leftmost, increases rightward)\n"
                "  - y = row index    (0 = topmost,  increases downward)\n"
                "  - ACTION6(x=col, y=row) clicks the cell at that position.\n\n"
                "=== AVAILABLE ACTIONS ===\n"
                "- RESET: Restart the current level from scratch.\n"
                "- ACTION1 to ACTION5, ACTION7: Discrete operations.\n"
                "- ACTION6(x, y): Coordinate click on the cell at column x, row y (0–63).\n\n"
                "=== GENERAL SPATIAL LAYOUT RULES & INTERFACE METAPHORS ===\n"
                "Before choosing an action, perform general spatial analysis on the grid Layout:\n"
                "1. Spatial Separation & Target Matching: Identify separate color patches/objects in the grid.\n"
                "   - Compare the main movable shape and the target marker: Are their orientations/angles different? Are their sizes/lengths different?\n"
                "2. Linear Slider Layout Metaphor:\n"
                "   - Feature: A continuous horizontal or vertical line of uniform color, with a pointer dot of a different color somewhere along it.\n"
                "   - Function: This is a slider interface. Clicking different positions along it usually extends, shrinks, or translates the controlled object.\n"
                "   - Operation Sensitivity: You must click coordinate points inside the slider line itself to slide it.\n"
                "3. Rotary Selector Layout Metaphor:\n"
                "   - Feature: A compact, symmetric cluster of pixels around a central point, or dial-like structures.\n"
                "   - Function: This is a rotary dial or switch interface. Clicking it usually rotates or flips the controlled object, changing its orientation.\n"
                "4. Control-Target Independence:\n"
                "   - Active controls (buttons, dials, sliders) are spatially separated from passive goal markers. Focus your actions strictly on coordinates within the active button, dial, or slider structures.\n\n"
                "=== PDCA LOOP INSTRUCTION ===\n"
                "For every step, you must explicitly construct and update your PDCA loop:\n"
                "  - Plan (P): Analyze shape alignment gaps (orientation difference, size difference). Plan an action to test if a specific control dial/slider changes the orientation or length of the main shape.\n"
                "  - Do (D): Output the chosen action (discrete or coordinate click).\n"
                "  - Check (C): Did the action cause a local button color toggle, or a global shape change? Does the current state decrease the alignment gap to the target?\n"
                "  - Act (A): Adjust strategy. Focus on controls that trigger global shape transformations.\n\n"
                "=== OUTPUT FORMAT (STRICT JSON ONLY) ===\n"
                f"{format_example}\n"
                f"{action_restriction}\n"
                "RULES: pdca_plan, pdca_check, and pdca_act MUST be in Traditional Chinese (繁體中文)."
            )

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content_text}
            ]

            prompt = active_processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = active_processor(text=prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            with torch.no_grad():
                outputs = active_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=0.0
                )

            input_len = inputs["input_ids"].shape[1]
            new_tokens = outputs[0, input_len:]
            response = active_processor.decode(new_tokens, skip_special_tokens=True)
            print(f"[{self.game_id}] Step {self.action_counter + 1} model output (Generated by {model_id}):")
            print(response.strip())

            # 10. Parse LLM response
            action = parse_action_response(response, available_actions)
            llm_action_tuple = self.game_action_to_action_tuple(action)
            action_desc = self.action_tuple_to_desc(llm_action_tuple)

            if llm_action_tuple in unexplored_actions:
                chosen_action_tuple = llm_action_tuple
                print(f"[Graph Explorer] Executing LLM recommended unexplored action: {action_desc}")
            else:
                chosen_action_tuple = None
                try:
                    json_match = re.search(r"(\{.*\})", response, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group(1))
                        candidates = data.get("candidate_control_coords", [])
                        for coords_pair in candidates:
                            if isinstance(coords_pair, list) and len(coords_pair) == 2:
                                cx, cy = int(coords_pair[0]), int(coords_pair[1])
                                cand_tuple = ("CLICK", (cx, cy))
                                if cand_tuple in unexplored_actions:
                                    chosen_action_tuple = cand_tuple
                                    print(f"[Graph Explorer] Prioritizing unexplored candidate coordinate from LLM: ACTION6(x={cx}, y={cy})")
                                    break
                except Exception:
                    pass

                if chosen_action_tuple is None:
                    chosen_action_tuple = unexplored_actions[0]
                    desc = self.action_tuple_to_desc(chosen_action_tuple)
                    print(f"[Graph Explorer] Fallback: LLM choice already explored/invalid. Executing next unexplored action: {desc}")

            # Re-convert chosen_action_tuple to GameAction
            action = self.action_tuple_to_game_action(chosen_action_tuple)
            action_desc = self.action_tuple_to_desc(chosen_action_tuple)

            # Record step in history
            step_record = {
                "action_desc": action_desc,
                "reasoning": getattr(action, "reasoning", "No reasoning provided"),
                "pdca_plan": getattr(action, "pdca_plan", ""),
                "pdca_check": getattr(action, "pdca_check", ""),
                "pdca_act": getattr(action, "pdca_act", ""),
                "prev_grid": current_grid,
                "prev_score": current_level,
                "outcome": None,
                "prompt": prompt,
                "response": response.strip() + active_processor.tokenizer.eos_token
            }
            self.action_history.append(step_record)
            if len(self.action_history) > 15:
                self.action_history.pop(0)

            if action.name == "ACTION6" and hasattr(action, "action_data") and action.action_data:
                data = action.action_data.model_dump()
                self.probed_coordinates.add((int(data.get("x", 0)), int(data.get("y", 0))))

            self.last_state_hash = current_hash
            self.last_action_taken = chosen_action_tuple
            return action

        except Exception as e:
            print(f"[DEBUG] choose_action CRASHED: {type(e).__name__}: {e}")
            traceback.print_exc()
            action = random.choice([
                GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                GameAction.ACTION4, GameAction.ACTION5
            ])
            action.reasoning = f"Fallback after exception: {e}"
            return action
