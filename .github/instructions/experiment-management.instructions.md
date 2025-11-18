---
description: 'Experiment tracking, artifacts, reproducibility for VLA research'
applyTo: '**/*experiment*.py, **/*config*.yaml, **/*config*.json'
---

# Experiment Management

Standards for tracking experiments and ensuring reproducibility.

## Configuration Logging

**Log ALL hyperparameters at start:**

```yaml
experiment:
  name: "rt2_baseline_v1"
  seed: 42
  
model:
  type: "RT2"
  vision_encoder: "ViT-B/16"
  num_params: "300M"
  
training:
  batch_size: 256
  learning_rate: 3.0e-4
  num_steps: 100000
  optimizer: "AdamW"
  
data:
  dataset: "CALVIN"
  augmentation: true
```

## Metrics Tracking

Use W&B, TensorBoard, or MLflow:

```python
import wandb

wandb.init(project="vla", name="exp_001", config=config)
wandb.log({"train/loss": loss, "epoch": epoch})
wandb.save("checkpoints/best.pt")
```

## Version Control

**Code**: Commit before experiments, record git hash
**Data**: Use DVC, record checksums
**Models**: Semantic versioning, metadata files

## Reproducibility Checklist

```markdown
- [ ] Python version: 3.9.12
- [ ] PyTorch version: 2.0.1
- [ ] CUDA version: 11.8
- [ ] Random seed set: 42
- [ ] Git commit: abc123def
- [ ] Dataset version: v2.1
- [ ] Hardware: 4× A100 40GB
```

## File Organization

```
experiments/
├── exp_001_baseline/
│   ├── README.md          # 描述 + 风险 + 结果
│   ├── config.yaml        # 完整配置
│   ├── train.py           # 训练代码
│   ├── checkpoints/       # 模型检查点
│   ├── logs/              # 训练日志
│   └── results/           # 评估结果
└── exp_002_improved/
    └── ...
```

## README Requirements

```markdown
# Experiment 001: RT-2 Baseline

## Description
Train RT-2 model on CALVIN dataset

## Research Goals
- Establish baseline performance
- Validate training pipeline
- Measure generalization

## ⚠️ Risk Warnings

### 🔴 Critical
- **Overwriting Results**: Running experiment again overwrites results/
  - Mitigation: Backup results/ before rerun
  
### 🟡 High
- **Storage Overflow**: Generates ~500GB checkpoints
  - Monitor: Disk usage logged
  - Cleanup: Use `--cleanup_old` to remove old checkpoints

## Results
- Success Rate: 78.5% (target: 75%)
- Training Time: 36 GPU-hours
- Final Loss: 0.0198

## Reproducibility
```bash
conda env create -f environment.yml
python train.py --config exp_001/config.yaml
```

Hardware: 4× RTX 3090
```

## Experiment Workflow

**Before:**
- [ ] Define hypothesis
- [ ] Plan experiment
- [ ] Check disk space
- [ ] Commit code
- [ ] Create exp folder + README

**During:**
- [ ] Monitor metrics
- [ ] Save checkpoints
- [ ] Log to tracking platform
- [ ] Check for anomalies

**After:**
- [ ] Generate plots
- [ ] Document findings
- [ ] Update README
- [ ] Archive artifacts
- [ ] Plan next iteration

## Integration Examples

**W&B:**
```python
wandb.init(project="vla", config=config)
wandb.log({"loss": loss})
```

**TensorBoard:**
```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('logs/')
writer.add_scalar('Loss/train', loss, step)
```

## Knowledge Management

- Maintain experiment database/spreadsheet
- Document dead ends to avoid repetition
- Share insights in team meetings
- Create summary docs for key findings
