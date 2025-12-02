"""
快速查看处理后的 MIDI -> piano-roll 数据格式。

功能：
1. 随机抽取若干样本，显示张量形状、激活密度（>0.5）、力度统计。
2. 打印前若干时间帧的 ASCII 可视化（# 代表激活）。
3. 可选择保存为 PNG 热力图与重新导出回 MIDI 验证一致性。

用法示例：
CUDA_VISIBLE_DEVICES=0 python inspect_processed.py \
  --midi_dir /mnt/ftpdata/tianming/Datasets/POP909_dataset \
  --num_samples 3 \
  --save_png_dir vis_png \
  --save_midi_dir vis_midi
"""
import os
import argparse
import random
import numpy as np
import torch
from midi_dataset import MidiRollDataset
from midi_processor_wrapper import save_roll_as_midi

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--midi_dir", type=str, required=True)
    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument("--time_frames", type=int, default=128, help="与训练时一致的裁剪长度")
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--save_png_dir", type=str, default="")
    p.add_argument("--save_midi_dir", type=str, default="")
    p.add_argument("--ascii_time", type=int, default=32, help="ASCII 可视化的时间帧数")
    p.add_argument("--ascii_pitch_low", type=int, default=48, help="ASCII 起始音高（MIDI号）")
    p.add_argument("--ascii_pitch_high", type=int, default=72, help="ASCII 结束音高（MIDI号）")
    return p.parse_args()


def to_float_tensor(x_u8: torch.Tensor) -> torch.Tensor:
    # uint8 [0,255] -> float [-1,1]（训练前的归一化）
    return x_u8.float() / 255.0 * 2.0 - 1.0


def ascii_vis(roll_01: np.ndarray, time_frames: int, pitch_low: int, pitch_high: int, threshold=0.5):
    """
    roll_01: [P, T]，数值 0..1
    仅展示选定 pitch 范围与前 time_frames 帧
    """
    P, T = roll_01.shape
    t_show = min(time_frames, T)
    lines = []
    for pitch in range(pitch_high, pitch_low - 1, -1):
        if not (0 <= pitch < P):
            continue
        row = roll_01[pitch, :t_show]
        chars = ["#" if v >= threshold else "." for v in row]
        lines.append(f"{pitch:03d} " + "".join(chars))
    return "\n".join(lines)


def main():
    args = build_args()
    ds = MidiRollDataset(
        midi_dir=args.midi_dir,
        fps=args.fps,
        time_frames=args.time_frames,
        random_crop=True
    )

    indices = random.sample(range(len(ds)), k=min(args.num_samples, len(ds)))

    if args.save_png_dir:
        os.makedirs(args.save_png_dir, exist_ok=True)
    if args.save_midi_dir:
        os.makedirs(args.save_midi_dir, exist_ok=True)

    for i, idx in enumerate(indices):
        x_u8, _ = ds[idx]              # [1, 128, T]
        roll_u8 = x_u8[0].numpy()      # [128, T]
        roll_01 = roll_u8.astype(np.float32) / 255.0  # [0,1]

        # 统计
        active = (roll_01 >= 0.5).sum()
        total = roll_01.size
        density = active / total
        vmax = roll_01.max()
        vmean = roll_01.mean()

        print(f"\n=== Sample {i} (index {idx}) ===")
        print(f"Shape (pitch x time): {roll_01.shape}")
        print(f"激活密度(>=0.5): {density:.4f}")
        print(f"力度 最大: {vmax:.3f} 均值: {vmean:.3f}")

        # ASCII 可视化 (子范围)
        vis_text = ascii_vis(
            roll_01,
            time_frames=args.ascii_time,
            pitch_low=args.ascii_pitch_low,
            pitch_high=args.ascii_pitch_high,
            threshold=0.5
        )
        print("ASCII Piano-Roll (上高下低，#=激活):")
        print(vis_text)

        # 保存 PNG
        if plt and args.save_png_dir:
            plt.figure(figsize=(10, 4))
            plt.imshow(roll_01, aspect="auto", origin="lower", interpolation="nearest", cmap="magma")
            plt.colorbar(label="Normalized Velocity (0..1)")
            plt.title(f"Sample {i}  roll (pitch x time)")
            plt.xlabel("Time Frame")
            plt.ylabel("Pitch")
            png_path = os.path.join(args.save_png_dir, f"sample_{i:02d}.png")
            plt.tight_layout()
            plt.savefig(png_path)
            plt.close()
            print(f"PNG 保存: {png_path}")
        elif not plt and args.save_png_dir:
            print("matplotlib 未安装，跳过 PNG 保存。可执行: pip install matplotlib")

        # 回写 MIDI 验证
        if args.save_midi_dir:
            pr_time_pitch = roll_01.T  # [T, P]
            midi_path = os.path.join(args.save_midi_dir, f"sample_{i:02d}.mid")
            save_roll_as_midi(pr_time_pitch, midi_path, fps=args.fps, min_pitch=0, velocity=100)
            print(f"MIDI 保存: {midi_path}")

    print("\n完成。")


if __name__ == "__main__":
    main()