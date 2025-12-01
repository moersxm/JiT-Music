import os
import argparse
import torch
import numpy as np

from denoiser_music import DenoiserMusic
from midi_processor_wrapper import save_roll_as_midi


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--velocity", type=int, default=100)
    return p.parse_args()


def main():
    """
    生成流程（适配非正方形 pitch_size × time_size）:
    1) 读取 ckpt，恢复训练时保存的参数（含 img_h=音高维, img_w=时间帧数）。
    2) 构建模型并替换为 EMA 权重（ema1）。
    3) 从高斯噪声出发依次时间步积分 (Euler/Heun)，内部进行 CFG。
    4) 输出张量 z: [B,1,img_h,img_w]（[-1,1]），映射到 [0,1] 得到 piano-roll:
       roll[b,0] 形状 [img_h, img_w] -> 转置为 [img_w, img_h] 即 [T, P] 写入 MIDI。
    """
    args = build_args()
    os.makedirs(args.save_dir, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    ns = argparse.Namespace(**ckpt["args"])
    img_h = getattr(ns, "img_h", 128)
    img_w = getattr(ns, "img_w", 128)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DenoiserMusic(ns).to(device).eval()
    model.load_state_dict(ckpt["model"])
    with torch.no_grad():
        model.ema_params1 = [p.detach().clone().to(device) for p in model.parameters()]
        model.ema_params2 = [p.detach().clone().to(device) for p in model.parameters()]
        ema_state = {k: v for k, v in model.state_dict().items()}
        for i, (name, _) in enumerate(model.named_parameters()):
            ema_state[name] = ckpt["ema1"][i].to(device)
        model.load_state_dict(ema_state)

    remain = args.num_samples
    bs = min(remain, 16)
    sid = 0

    while remain > 0:
        cur = min(remain, bs)
        labels = torch.zeros(cur, dtype=torch.long, device=device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model.generate(labels)  # [B,1,img_h,img_w]
        roll = (out + 1) / 2.0
        roll = roll.clamp(0, 1).detach().cpu().numpy()

        for b in range(cur):
            # roll[b,0]: [img_h, img_w] -> MIDI 需要 [T, P] = [time, pitch]
            pr = np.transpose(roll[b, 0], (1, 0))  # [img_w, img_h]
            midi_path = os.path.join(args.save_dir, f"sample_{sid:04d}.mid")
            save_roll_as_midi(pr, midi_path, fps=args.fps, min_pitch=0, velocity=args.velocity)
            sid += 1
        remain -= cur

    print(f"saved {args.num_samples} midi files to {args.save_dir} (shape pitch={img_h}, time={img_w})")


if __name__ == "__main__":
    main()