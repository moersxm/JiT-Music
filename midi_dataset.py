import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from midi_processor_wrapper import MidiToPianoRoll


class MidiRollDataset(Dataset):
    def __init__(self, midi_dir: str, fps: int = 16, time_frames: int = 128,
                 min_pitch: int = 0, max_pitch: int = 127, random_crop: bool = True):
        self.files = sorted([p for ext in ("*.mid", "*.midi") for p in glob.glob(os.path.join(midi_dir, "**", ext), recursive=True)])
        if len(self.files) == 0:
            raise RuntimeError(f"未在 {midi_dir} 找到 .mid/.midi 文件")
        self.proc = MidiToPianoRoll(fps=fps, min_pitch=min_pitch, max_pitch=max_pitch)
        self.time_frames = time_frames
        self.random_crop = random_crop

    def __len__(self):
        return len(self.files)

    def _fit_time(self, roll: np.ndarray) -> np.ndarray:
        T = roll.shape[0]
        target = self.time_frames
        if T == target:
            return roll
        if T < target:
            pad = target - T
            left = 0 if not self.random_crop else random.randint(0, pad)
            right = pad - left
            return np.pad(roll, ((left, right), (0, 0)), mode="constant")
        # T > target
        start = 0 if not self.random_crop else random.randint(0, T - target)
        return roll[start:start + target]

    def __getitem__(self, idx: int):
        path = self.files[idx]
        roll = self.proc.midi_to_roll(path)  # [T, 128]
        roll = self._fit_time(roll)
        roll_u8 = np.clip(roll * 255.0 + 0.5, 0, 255).astype(np.uint8)
        x = torch.from_numpy(roll_u8).permute(1, 0).unsqueeze(0).contiguous()  # [1, 128, 128]
        y = torch.tensor(0, dtype=torch.long)  # 单类
        return x, y