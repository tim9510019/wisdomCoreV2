import wave
import struct
import numpy as np

def generate_melody_wav(filename="music.wav", sr=16000):
    print("Generating synthetic music melody...")
    # Define note frequencies for a simple melody (A major chord notes)
    notes = [440.0, 554.37, 659.25, 880.0]  # A4, C#5, E5, A5
    durations = [0.5, 0.5, 0.5, 1.0]       # durations in seconds
    
    waveform = []
    for freq, dur in zip(notes, durations):
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        # Generate sine wave for each note with a slight amplitude envelope to sound natural
        envelope = np.sin(np.pi * np.linspace(0, 1, len(t)))  # simple half-sine envelope
        note_wave = 0.4 * np.sin(2 * np.pi * freq * t) * envelope
        waveform.extend(note_wave)
        
    waveform = np.array(waveform, dtype=np.float32)
    
    # Write to WAV file using built-in wave module
    print(f"Saving melody to {filename}...")
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sr)
        for sample in waveform:
            # Quantize float32 [-1, 1] to int16
            val = int(max(-32768, min(32767, sample * 32767)))
            data = struct.pack('<h', val)
            wav_file.writeframesraw(data)
    print("Done!")

if __name__ == "__main__":
    generate_melody_wav()
