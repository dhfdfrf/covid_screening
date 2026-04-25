# RL Dynamic Fusion for QaTa-COV19 CXR Infection Segmentation

这个项目实现了一个适合你当前需求的最小科研原型：

- 三个分割主干：`TransUNet2D` / `Swin-Unet2D` / `UCTransNet2D`
- 一个 PPO 风格策略网络
- 对三条分支输出进行动态加权融合
- 奖励函数以 `Dice增益 + IoU项` 为核心
- 适用于二分类胸部 X 光感染区域分割

## 目录结构

```text
rl_seg_project/
├── src/
│   ├── data/dataset.py
│   ├── models/
│   ├── rl/
│   └── utils/
├── configs/
├── train_base_models.py
├── train_rl_fusion.py
├── infer.py
└── requirements.txt
```

## 数据组织

```text
dataset/
├── images/
│   ├── xxx.png
│   └── ...
└── masks/
    ├── xxx.png
    └── ...
```

要求图像与 mask 同名。

## 训练流程

### 1) 先训练三个主干

```bash
python train_base_models.py --model transunet2d --image_dir dataset/images --mask_dir dataset/masks
python train_base_models.py --model swin_unet2d --image_dir dataset/images --mask_dir dataset/masks
python train_base_models.py --model uctransnet2d --image_dir dataset/images --mask_dir dataset/masks
```

### 2) 再训练 RL 融合策略

```bash
python train_rl_fusion.py \
  --image_dir dataset/images \
  --mask_dir dataset/masks \
  --trans_ckpt checkpoints/base/transunet2d_best.pt \
  --swin_ckpt checkpoints/base/swin_unet2d_best.pt \
  --uct_ckpt checkpoints/base/uctransnet2d_best.pt
```

### 3) 推理

```bash
python infer.py \
  --image dataset/images/sample.png \
  --trans_ckpt checkpoints/base/transunet2d_best.pt \
  --swin_ckpt checkpoints/base/swin_unet2d_best.pt \
  --uct_ckpt checkpoints/base/uctransnet2d_best.pt \
  --rl_ckpt checkpoints/rl/rl_fusion_best.pt \
  --save_path pred.png
```

## 第一性原理说明

这个方案把问题拆成两层：

1. **像素级建模**：交给分割网络做，它们擅长从图像到 mask 的复杂映射。
2. **样本级决策**：交给强化学习做，它擅长“当前这张图更该信哪一个模型”。

所以 RL 不直接替代分割网络，而是学习一个更高层的策略：

- 病灶边界模糊时，可能更依赖某个 Transformer 分支。
- 小区域病灶时，可能更依赖另一个对局部更敏感的分支。
- 普通情况则学习近似平均，但不是固定平均。

## 说明

- 为了兼顾 4060 8GB 和 <10 小时约束，这个版本默认**冻结主干，只训练融合策略**。
- `SwinUNETR` 如果本地 MONAI 版本不兼容，会自动退化到一个小型 fallback U-Net，保证项目仍可跑通。
- 这个 PPO 是为你的单步融合决策专门写的轻量版，不依赖 SB3。

## 你后续最值得做的两件事

1. 增加 `k-fold`、对比平均融合、对比单模型结果。
2. 把 reward 从单纯 Dice 增益升级成 `Dice + Boundary F1 + sparsity penalty`。
