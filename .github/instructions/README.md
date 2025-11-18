# Instructions Directory

Specialized instruction files for GitHub Copilot, targeting specific file types.

## File Pattern System

| File | Pattern | Purpose |
|------|---------|---------|
| `dataset-processing` | `**/*dataset*.py, **/*data*.py` | Data processing & validation |
| `model-training` | `**/*train*.py, **/*model*.py` | Training & monitoring |
| `model-evaluation` | `**/*eval*.py, **/*test*.py` | Evaluation & metrics |
| `experiment-management` | `**/*experiment*.py, **/*config*.yaml` | Tracking & reproducibility |
| `ai-prompt-engineering` | `**/*` | General AI best practices |

## Structure

Each file follows:

```yaml
---
description: 'Brief purpose (1-500 chars)'
applyTo: 'glob pattern'
---

# Title
Overview

## General Instructions
High-level guidelines

## Best Practices
Tables & Good/Bad examples

## Common Patterns
Code templates

## Validation
Testing & checklist
```

## How It Works

```
Create: train_model.py
   ↓
Match: **/*train*.py
   ↓
Load: model-training.instructions.md
   ↓
Apply: Training standards & patterns
```

## Writing Guidelines

**✅ Do:**
- Clear imperative language
- Specific, actionable items
- Good vs Bad code examples
- Tables for structure

**❌ Don't:**
- Vague terms
- Long paragraphs
- Missing examples
