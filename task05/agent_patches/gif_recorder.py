import os
import time
import numpy as np
from PIL import Image

# ARC-AGI 經典 10 色調色盤 (RGB)
ARC_COLOR_MAP = [
    (0, 0, 0),        # 0: black
    (0, 116, 217),    # 1: blue
    (255, 65, 54),    # 2: red
    (46, 204, 64),    # 3: green
    (255, 220, 0),    # 4: yellow
    (170, 170, 170),  # 5: gray
    (240, 18, 190),   # 6: magenta
    (255, 133, 27),   # 7: orange
    (127, 219, 255),  # 8: teal
    (135, 12, 37)     # 9: maroon
]

# 全域時間戳，同一次執行共用同一個資料夾
_run_timestamp = time.strftime("%Y%m%d_%H%M%S")
_workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RECORDINGS_ROOT = os.path.join(_workspace_root, "recordings", f"run_{_run_timestamp}")

class GIFRecorder:
    def __init__(self, game_id: str, agent_name: str, scale: int = 10):
        self.game_id = game_id
        # 移除非法字元以作為資料夾名稱的一部分
        safe_agent_name = "".join([c if c.isalnum() or c in "-_" else "_" for c in agent_name])
        self.agent_name = safe_agent_name
        self.scale = scale
        
        self.run_dir = RECORDINGS_ROOT
        self.game_dir = os.path.join(self.run_dir, f"{game_id}_{self.agent_name}")
        
        os.makedirs(self.game_dir, exist_ok=True)
        self.frames = []
        self.step_counter = 0
        print(f"[GIFRecorder] Initialized for {self.game_id} (Agent: {self.agent_name}). Frames will be saved to: {self.game_dir}")

    def record_frame(self, grid) -> None:
        """將 2D grid 轉換為放大後的 RGB 圖片並儲存。"""
        if grid is None:
            return
        try:
            grid_arr = np.array(grid, dtype=np.int32)
            if grid_arr.ndim != 2:
                # 確保是 2D array
                return
            h, w = grid_arr.shape
            
            # 建立一個放大後的 RGB 圖像
            img_h, img_w = h * self.scale, w * self.scale
            img_data = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            
            for r in range(h):
                for c in range(w):
                    color_idx = grid_arr[r, c]
                    color_idx = max(0, min(9, int(color_idx)))
                    color_rgb = ARC_COLOR_MAP[color_idx]
                    img_data[r*self.scale : (r+1)*self.scale, c*self.scale : (c+1)*self.scale] = color_rgb
                    
            img = Image.fromarray(img_data, 'RGB')
            
            # 儲存 PNG 檔案
            filename = os.path.join(self.game_dir, f"step_{self.step_counter:04d}.png")
            img.save(filename)
            
            # 保存在記憶體中以便後續合成 GIF
            self.frames.append(img)
            self.step_counter += 1
        except Exception as e:
            print(f"[GIFRecorder] Error recording frame for {self.game_id}: {e}")

    def save_gif(self, duration: int = 300) -> None:
        """將所有已記錄的 frames 合成 GIF 儲存。"""
        if not self.frames:
            print(f"[GIFRecorder] No frames to save for {self.game_id}")
            return
        try:
            gif_filename = os.path.join(self.run_dir, f"{self.game_id}_{self.agent_name}.gif")
            
            # 合成 GIF
            self.frames[0].save(
                gif_filename,
                save_all=True,
                append_images=self.frames[1:],
                optimize=False,
                duration=duration,
                loop=0
            )
            print(f"[GIFRecorder] GIF successfully saved to: {gif_filename}")
        except Exception as e:
            print(f"[GIFRecorder] Error saving GIF for {self.game_id}: {e}")
