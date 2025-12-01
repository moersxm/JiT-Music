import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

# 复用 JiT 的训练引擎
JIT_ROOT = "/home/tianming/code/JiT"
if JIT_ROOT not in sys.path:
    sys.path.insert(0, JIT_ROOT)
from engine_jit import train_one_epoch  # type: ignore

from midi_dataset import MidiRollDataset
from denoiser_music import DenoiserMusic

def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--midi_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    # 现在拆分成 pitch_size 与 time_size，pitch_size 默认为 128（音高数），time_size 可加长
    p.add_argument("--pitch_size", type=int, default=128)
    p.add_argument("--time_size", type=int, default=1024)

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--log_freq", type=int, default=50)
    p.add_argument("--lr_schedule", type=str, default="cosine", choices=["cosine", "constant"])

    p.add_argument("--model", type=str, default="JiT-B/16")
    p.add_argument("--class_num", type=int, default=1)
    p.add_argument("--attn_dropout", type=float, default=0.0)
    p.add_argument("--proj_dropout", type=float, default=0.0)

    p.add_argument("--label_drop_prob", type=float, default=0.1)
    p.add_argument("--P_mean", type=float, default=-1.2)
    p.add_argument("--P_std", type=float, default=1.2)
    p.add_argument("--t_eps", type=float, default=1e-3)
    p.add_argument("--noise_scale", type=float, default=1.0)
    p.add_argument("--ema_decay1", type=float, default=0.999)
    p.add_argument("--ema_decay2", type=float, default=0.9999)

    p.add_argument("--sampling_method", type=str, default="heun")
    p.add_argument("--num_sampling_steps", type=int, default=32)
    p.add_argument("--cfg", type=float, default=3.0)
    p.add_argument("--interval_min", type=float, default=0.0)
    p.add_argument("--interval_max", type=float, default=1.0)

    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_epochs", type=int, default=2)
    return p.parse_args()


def main():
    args = build_args()
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 训练长序列时使用 time_size
    ds = MidiRollDataset(midi_dir=args.midi_dir, time_frames=args.time_size, random_crop=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True)

    # 传入 (pitch_size, time_size)
    args.img_h = args.pitch_size
    args.img_w = args.time_size

    model = DenoiserMusic(args).to(device)
    model_without_ddp = model

    params = list(model.parameters())
    model.ema_params1 = [p.detach().clone() for p in params]
    model.ema_params2 = [p.detach().clone() for p in params]

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    for epoch in range(args.epochs):
        train_one_epoch(model, model_without_ddp, dl, optimizer, device, epoch, log_writer=None, args=args)
        state = {
            "model": model.state_dict(),
            "ema1": [p.detach().cpu() for p in model.ema_params1],
            "ema2": [p.detach().cpu() for p in model.ema_params2],
            "args": vars(args),
            "epoch": epoch,
        }
        torch.save(state, os.path.join(ckpt_dir, "last.pt"))
        if (epoch + 1) % 50 == 0:   # 每 50 轮保存一次
            torch.save(state, os.path.join(ckpt_dir, f"epoch_{epoch+1}.pt"))
    print("done")


if __name__ == "__main__":
    main()