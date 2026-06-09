import os
import sys
import wave
import torch
import numpy as np
import av
from PIL import Image, ImageDraw

# Add parent directory to sys.path to resolve GEMMA4 imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from GEMMA4 import GEMMA4, GEMMA4Processor

def color_print(text, color="green"):
    colors = {
        "green": "\033[92m",
        "blue": "\033[94m",
        "cyan": "\033[96m",
        "yellow": "\033[93m",
        "magenta": "\033[95m",
        "bold": "\033[1m",
        "end": "\033[0m"
    }
    print(f"{colors.get(color, '')}{text}{colors['end']}")

def load_wav_mono_normalized(filename):
    """
    Loads a 16-bit PCM WAV file using built-in wave module and normalizes it to [-1.0, 1.0].
    """
    with wave.open(filename, 'rb') as wav_file:
        n_channels = wav_file.getnchannels()
        sampwidth = wav_file.getsampwidth()
        framerate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        
        raw_data = wav_file.readframes(n_frames)
        if sampwidth == 2:
            data = np.frombuffer(raw_data, dtype=np.int16)
        else:
            raise ValueError(f"Only 16-bit PCM WAV files are supported. Found width: {sampwidth}")
            
        waveform = data.astype(np.float32) / 32768.0
        
        # If stereo, mix down to mono
        if n_channels > 1:
            waveform = waveform.reshape(-1, n_channels).mean(axis=1)
            
        return waveform, framerate

def generate_mp4_video(filename, width=224, height=224, num_frames=32, fps=24):
    """
    Compiles 32 frames of a moving red circle into a physical MP4 video file using PyAV.
    """
    container = av.open(filename, mode='w')
    stream = container.add_stream('h264', rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = 'yuv420p'
    
    for i in range(num_frames):
        img = Image.new("RGB", (width, height), color=(10, 10, 50))
        draw = ImageDraw.Draw(img)
        r = 20
        cx = int(30 + (width - 60) * (i / (num_frames - 1)))
        cy = height // 2
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(220, 20, 20))
        
        frame = av.VideoFrame.from_image(img)
        for packet in stream.encode(frame):
            container.mux(packet)
            
    # Flush video stream
    for packet in stream.encode():
        container.mux(packet)
        
    container.close()

def main():
    color_print("==========================================================", "bold")
    color_print("🚀 原生 Gemma 4 統一多模態推理與自迴歸生成展示 DEMO", "bold")
    color_print("==========================================================", "bold")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    color_print(f"🖥  執行硬體裝置: {device}", "cyan")
    
    # 1. 初始化 Processor
    color_print("\n📥 正在初始化 Gemma 4 Multimodal Processor...", "yellow")
    processor = GEMMA4Processor()
    
    # 2. 載入原生 Gemma 4 2B 模型
    color_print("\n⚙️  正在載入 Gemma 4 官方模型與權重...", "yellow")
    model = GEMMA4.load_model(device_map="cuda")
    model.eval()
    color_print("✅ Gemma 4 模型載入完成！", "green")
    
    # ==========================================
    # 展示 1：純文字推理 (驗證 Gemma 4 原生知識能力)
    # ==========================================
    color_print("\n📌 展示 1：純文字推理 (驗證 Gemma 4 知識回答)", "magenta")
    
    messages_text = [
        {
            "role": "user", 
            "content": [{"type": "text", "text": "What is the capital city of France? Answer in one word."}]
        }
    ]
    prompt_text = processor.apply_chat_template(messages_text, add_generation_prompt=True)
    color_print(f"[Prompt Input]: {prompt_text.strip()}", "blue")
    
    inputs_text = processor(text=prompt_text).to(device)
    
    with torch.no_grad():
        outputs_text = model.generate(**inputs_text, max_new_tokens=15)
        
    response_text = processor.decode(outputs_text[0], skip_special_tokens=True)
    model_response_text = response_text.split("model\n")[-1] if "model\n" in response_text else response_text
    color_print(f"[Generated Output]: {model_response_text.strip()}", "green")
    print("-" * 50)
    
    # ==========================================
    # 展示 2：影像多模態推理 (影像特徵理解)
    # ==========================================
    color_print("\n📌 展示 2：多模態影像推理 (輸入 Eiffel Tower 照片)", "magenta")
    
    image_path = os.path.join(current_dir, "eiffel_tower.png")
    if not os.path.exists(image_path):
        color_print(f"❌ 錯誤：找不到測試圖片 {image_path}", "magenta")
        sys.exit(1)
        
    color_print(f"📷 正在載入測試影像: {image_path}", "cyan")
    image = Image.open(image_path)
    
    messages_image = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What landmark is shown in this image and where is it located?"}
            ]
        }
    ]
    
    prompt_image = processor.apply_chat_template(messages_image, add_generation_prompt=True)
    color_print(f"[Prompt Input]: <|image|>{prompt_image.split('<|image|>')[-1].strip()}", "blue")
    
    inputs_image = processor(text=prompt_image, images=image).to(device)
    
    with torch.no_grad():
        outputs_image = model.generate(**inputs_image, max_new_tokens=50)
        
    response_image = processor.decode(outputs_image[0], skip_special_tokens=True)
    model_response_image = response_image.split("model\n")[-1] if "model\n" in response_image else response_image
    color_print(f"[Generated Output]:\n{model_response_image.strip()}", "green")
    print("-" * 50)

    # ==========================================
    # 展示 3：音訊多模態推理 (聽音樂與音效特徵理解)
    # ==========================================
    color_print("\n📌 展示 3：多模態音訊推理 (載入實體音樂檔案 music.wav)", "magenta")
    
    audio_path = os.path.join(current_dir, "music.wav")
    if not os.path.exists(audio_path):
        color_print(f"❌ 錯誤：找不到測試音訊 {audio_path}", "magenta")
        sys.exit(1)
        
    color_print(f"🎵 正在自硬碟載入音訊檔案: {audio_path}", "cyan")
    waveform, sr = load_wav_mono_normalized(audio_path)
    color_print(f"   音訊取樣率: {sr} Hz | 總樣本數: {len(waveform)}", "cyan")
    
    messages_audio = [
        {
            "role": "user",
            "content": [
                {"type": "audio"},
                {"type": "text", "text": "Describe what you hear in this audio clip. What kind of sound is it?"}
            ]
        }
    ]
    
    prompt_audio = processor.apply_chat_template(messages_audio, add_generation_prompt=True)
    color_print(f"[Prompt Input]: <|audio|>{prompt_audio.split('<|audio|>')[-1].strip()}", "blue")
    
    inputs_audio = processor(text=prompt_audio, audio=waveform, sampling_rate=sr).to(device)
    
    with torch.no_grad():
        outputs_audio = model.generate(**inputs_audio, max_new_tokens=60)
        
    response_audio = processor.decode(outputs_audio[0], skip_special_tokens=True)
    model_response_audio = response_audio.split("model\n")[-1] if "model\n" in response_audio else response_audio
    color_print(f"[Generated Output]:\n{model_response_audio.strip()}", "green")
    print("-" * 50)

    # ==========================================
    # 展示 4：影片多模態推理 (影片特徵理解)
    # ==========================================
    color_print("\n📌 展示 4：多模態影片推理 (實體影片檔案 video.mp4)", "magenta")
    
    video_path = os.path.join(current_dir, "video.mp4")
    color_print(f"🎬 正在合成並建立實體 MP4 影片檔案: {video_path}", "cyan")
    generate_mp4_video(video_path, width=224, height=224, num_frames=32, fps=24)
    color_print("   影片檔案寫入完成！", "cyan")
    
    messages_video = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": "Describe the objects and colors shown in this video file."}
            ]
        }
    ]
    
    prompt_video = processor.apply_chat_template(messages_video, add_generation_prompt=True)
    color_print(f"[Prompt Input]: <|video|>{prompt_video.split('<|video|>')[-1].strip()}", "blue")
    
    inputs_video = processor(text=prompt_video, videos=video_path).to(device)
    
    with torch.no_grad():
        outputs_video = model.generate(**inputs_video, max_new_tokens=60)
        
    response_video = processor.decode(outputs_video[0], skip_special_tokens=True)
    model_response_video = response_video.split("model\n")[-1] if "model\n" in response_video else response_video
    color_print(f"[Generated Output]:\n{model_response_video.strip()}", "green")
    print("-" * 50)

    # ==========================================
    # 展示 5：三模態故事編寫 (圖片 + 聲音 + 影片 $\rightarrow$ 編故事)
    # ==========================================
    color_print("\n📌 展示 5：多模態合流故事編寫 (圖片 + 聲音 + 影片 ➡️ 繁體中文編故事)", "magenta")
    
    messages_story = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "audio"},
                {"type": "video"},
                {"type": "text", "text": "Weave a short, creative story in Traditional Chinese that combines the landmark from the image, the sound from the audio, and the motion from the video."}
            ]
        }
    ]
    
    prompt_story = processor.apply_chat_template(messages_story, add_generation_prompt=True)
    color_print(f"[Prompt Input]: <|image|><|audio|><|video|>{prompt_story.split('<|video|>')[-1].strip()}", "blue")
    
    # 傳入所有多模態輸入進行特徵提取
    inputs_story = processor(
        text=prompt_story, 
        images=image, 
        audio=waveform, 
        sampling_rate=sr, 
        videos=video_path
    ).to(device)
    
    # 提取 input_ids 避免 generation utils 中 shape 提取問題
    input_ids_story = inputs_story.pop("input_ids")
    
    with torch.no_grad():
        outputs_story = model.generate(input_ids_story, **inputs_story, max_new_tokens=250)
        
    response_story = processor.decode(outputs_story[0], skip_special_tokens=True)
    model_response_story = response_story.split("model\n")[-1] if "model\n" in response_story else response_story
    color_print(f"[Generated Output]:\n{model_response_story.strip()}", "green")
    
    color_print("==========================================================", "bold")

if __name__ == "__main__":
    main()
