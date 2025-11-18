---
description: 'VLA model evaluation: metrics, visualization, benchmarking'
applyTo: '**/*eval*.py, **/*test*.py, **/*metric*.py, **/*benchmark*.py'
---

# Model Evaluation

Standards for evaluating Vision-Language-Action models.

## Standard Metrics

- **Success Rate**: % of completed tasks (with confidence intervals)
- **Action Accuracy**: MSE/L1 for continuous, accuracy for discrete
- **Language Grounding**: Attention alignment, semantic similarity
- **Trajectory Similarity**: DTW distance, Fréchet distance
- **Generalization**: Zero-shot, few-shot, cross-domain

## Domain-Specific

**Manipulation**: Grasp success, placement accuracy, smoothness
**Navigation**: SPL, collision rate, goal accuracy
**Multi-Task**: Per-task breakdown, task switching overhead

## Progress Tracking

```python
# 必须显示：当前进度 + 实时指标 + ETA
# Evaluating: 450/1000 (45%) | Success: 75.6% | ETA: 1h 52m
```

## Code Pattern

```python
from tqdm import tqdm

def evaluate_model(model, test_loader):
    """评估模型，带进度跟踪"""
    results = []
    
    with tqdm(total=len(test_loader), desc="Evaluating") as pbar:
        for batch in test_loader:
            with torch.no_grad():
                pred = model(**batch)
                metrics = compute_metrics(pred, batch['labels'])
                results.append(metrics)
            
            # 实时更新
            current_success = np.mean([r['success'] for r in results])
            pbar.set_postfix({'success_rate': f"{current_success:.1%}"})
            pbar.update(1)
    
    return aggregate_results(results)
```

## File Organization

```
evaluation_task/
├── README.md          # 评估设置 + 风险 + 预期结果
├── evaluate.py        # 评估脚本
├── metrics.py         # 指标计算
└── visualize.py       # 可视化
```

## README Requirements

```markdown
## Evaluation Setup
- Checkpoint: `checkpoints/best_model.pt` (100K steps)
- Dataset: CALVIN validation (1000 episodes)
- Metrics: Success rate, Action MSE, DTW

## ⚠️ Risk Warnings

### 🟡 High
- **Long Evaluation**: 2-4 hours for full run
  - Mitigation: Batch evaluation, checkpointing
  
- **Simulation Crashes**: May crash on edge cases
  - Recovery: Skip failed, continue

### 🟢 Medium
- **Disk Space**: Videos need ~20GB
  - Control: `--save_videos` flag optional

## Expected Results
- Success Rate: 75-80% (baseline: 65%)
- Action MSE: 0.015-0.020 (baseline: 0.025)

## Output Files
- `results/metrics/summary.json`
- `results/plots/success_by_task.png`
- `results/videos/failure_cases/`
```

## Visualization

- Attention maps for vision-language grounding
- Action prediction rollouts vs ground truth
- Learning curves and training dynamics
- Success/failure case videos
- Confusion matrices for classification

## Workflow Checklist

Before:
- [ ] Checkpoint validated
- [ ] Test dataset ready
- [ ] Metrics implemented
- [ ] README with risks

During:
- [ ] Progress displayed
- [ ] Metrics logging
- [ ] No crashes
- [ ] Memory stable

After:
- [ ] All metrics computed
- [ ] Visualizations generated
- [ ] Failures documented
- [ ] Baseline comparison done
