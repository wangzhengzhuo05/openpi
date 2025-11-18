#!/bin/bash

CMD_FIRST="XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_calvin_scratch --exp-name=calvin_full --overwrite"
CMD="XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_calvin_scratch --exp-name=calvin_full --resume"

echo "=============================="
echo "Starting training at $(date)"
echo "=============================="
eval $CMD_FIRST
echo "Training stopped at $(date)"
echo "Will restart in 5 seconds..."
sleep 5

while true
do
    echo "=============================="
    echo "Starting training at $(date)"
    echo "=============================="

    # 运行一次训练
    eval $CMD

    echo "Training stopped at $(date)"
    echo "Will restart in 5 seconds..."
    sleep 5
done
