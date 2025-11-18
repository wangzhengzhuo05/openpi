---
description: 'VLA model training: architecture, distributed training, monitoring'
applyTo: '**/*train*.py, **/*model*.py, **/*trainer*.py'
---

# Model Training

Standards for training Vision-Language-Action models.

## Architecture Selection

| Type | Use Case | Key Components |
|------|----------|----------------|
| Transformer (RT-1/2, Octo) | General VLA | Vision + Language + Action decoder |
| Diffusion (pi0, Diffusion Policy) | Precise actions | Diffusion model |
| Flow Matching (pi0) | Efficient sampling | Flow network |
| Generative (OpenVLA, PALM-E) | Large-scale multi-task | LLM + Vision tower |
| RL (PPO, SAC) | Online learning | Policy + Value network |

## Best Practices

| Practice | Implementation | Avoid |
|----------|----------------|-------|
| Gradient clipping | `clip_grad_norm_(params, 1.0)` | No clipping → NaN |
| LR warmup | Linear 2K steps | Full LR → instability |
| Checkpointing | Every N steps + best | Only final |
| Progress tracking | tqdm with ETA | Silent training |

## Code Example

**✅ Good - With Monitoring:**

```python
from tqdm import tqdm
import torch

def train_epoch(model, loader, optimizer, epoch):
    """训练一个epoch，带监控"""
    model.train()
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        loss = model(**batch).loss
        
        # 检查NaN
        if torch.isnan(loss):
            logger.error(f"⚠️ NaN at step {step}")
            save_checkpoint("nan_debug.pt")
            raise ValueError("NaN detected")
        
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f"{loss:.4f}",
            'grad': f"{grad_norm:.3f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
        })
```

**❌ Bad - No Monitoring:**

```python
def train(model, data_loader):
    for batch in data_loader:  # 无进度显示
        loss = model(batch)
        loss.backward()  # 无梯度裁剪
        optimizer.step()  # 无NaN检查
```

## File Organization

Multi-file tasks:
```
training_task/
├── README.md          # 必须：任务说明 + 风险警告
├── train.py           # 训练脚本
├── model.py           # 模型定义
└── config.yaml        # 配置
```

## README Requirements

```markdown
## ⚠️ Risk Warnings

### 🔴 Critical
- **GPU Hours**: 48h on 4× A100 (~$200 cloud cost)
- **Checkpoint Overwrite**: Will overwrite existing checkpoints
  - Mitigation: Backup before run

### 🟡 High
- **OOM Risk**: Large batch may exceed 24GB VRAM
  - Monitor: GPU memory logged
  - Recovery: Reduce batch size

## Resource Requirements
- Hardware: 4× RTX 3090+ (24GB)
- Disk: 100GB dataset + 500GB checkpoints
- Time: ~48 GPU-hours
```

## Progress Tracking

```python
# 必须显示：进度 + 指标 + ETA + 资源
# Epoch 12/100: [████░░░░] 40% | Loss: 0.023 ↓ | ETA: 8h 30m | GPU: 92%
```

## Workflow Checklist

Before training:
- [ ] Dataset validated
- [ ] Architecture configured
- [ ] Hyperparameters logged
- [ ] Checkpoint dir has space
- [ ] README with risk warnings

During:
- [ ] Loss decreasing
- [ ] No NaN values
- [ ] GPU >80% utilized
- [ ] Checkpoints saving

After:
- [ ] Best checkpoint identified
- [ ] Metrics documented
- [ ] Risks encountered logged
