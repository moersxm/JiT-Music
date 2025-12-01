# JiT-Music

概述
- 目标：将 JiT（Just image Transformer）的扩散式训练范式用于音乐生成。
- 表示：把 MIDI 转换为 piano-roll（pitch×time = 128×128），作为 1 通道“图像”（形状 [1, 128, 128]）。
- 模型：保留 JiT 的主体结构（PatchEmbed → 多层 Transformer Block + AdaLN → FinalLayer → Unpatchify），输入/输出为单通道。时刻 t 和类别 y 通过 Timestep/Label Embedding 注入，并用 AdaLN 做条件调制。
- 训练：v 预测目标，与原 JiT 的图像扩散一致：z = t x + (1-t) e，监督 v = (x - z)/(1-t)。
- 生成：使用 Heun/Euler ODE 步进，从噪声逐步得到 piano-roll，再转回 MIDI。

数据处理
- 使用 pretty_midi 将 .mid/.midi 转为 0..1 的 piano-roll（帧率 fps 默认 16），并裁剪/填充到固定 128 帧。
- 数据张量为 uint8 的 [1, 128, 128]，训练时在 engine 中归一化到 [-1, 1]。

模型结构
- PatchEmbed：将单通道 piano-roll 切为 p×p patch（默认 p=16），通过 1×1 升维到隐藏维度。
- Positional Encoding：2D sin-cos 固定位置编码。
- 条件注入：Timestep Embedding（sin-cos → MLP到 hidden），Label Embedding（num_classes+1，支持 CFG 的空标签），相加得到 c。Block 内通过 AdaLN 生成 shift/scale/gate 进行调制。
- Transformer Block：RMSNorm → Multi-Head Attention（可选 q/k 归一化）→ 残差；RMSNorm → SwiGLU-FFN → 残差；两支均由 AdaLN 控制。
- FinalLayer：RMSNorm + AdaLN → 线性映射到每个 patch 的像素并 Unpatchify 回图像。
- 输出：预测 x̂，用于计算 v̂ = (x̂ - z)/(1-t)。

环境
- 使用现有环境：conda activate jit_py311
- 需要 pretty_midi：pip install pretty_midi

训练
- 单卡示例：
  CUDA_VISIBLE_DEVICES=0 python /home/tianming/code/JiT-Music/train_music.py \
    --midi_dir /path/to/midi \
    --output_dir /home/tianming/code/JiT-Music/runs/jit_music

- 关键信息：
  - 输入/输出分辨率为 128×128（可在脚本内调整）。
  - 默认无条件（class_num=1），通过 label_drop_prob 实现 CFG 所需的空标签。
  - 训练循环直接复用 /home/tianming/code/JiT/engine_jit.py 的 train_one_epoch（AMP、LR 调度、EMA 逻辑一致）。

生成
- 从检查点采样并保存为 MIDI：
  CUDA_VISIBLE_DEVICES=0 python /home/tianming/code/JiT-Music/generate_music.py \
    --ckpt /home/tianming/code/JiT-Music/runs/jit_music/checkpoints/last.pt \
    --save_dir /home/tianming/code/JiT-Music/runs/jit_music/samples

文件说明
- model_music_jit.py：完整的音乐版 JiT 模型（自包含）。
- denoiser_music.py：扩散训练/采样包装，接口与原 Denoiser 对齐（但通道=1）。
- midi_processor_wrapper.py：MIDI 与 piano-roll 互转（基于 pretty_midi）。
- midi_dataset.py：从目录读取 .mid/.midi，转为 [1, 128, 128] 张量。
- train_music.py：复用 JiT 的训练引擎进行训练。
- generate_music.py：用保存的 EMA 权重采样并写出 MIDI。# JiT-Music
