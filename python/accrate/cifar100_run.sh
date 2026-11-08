#!/bin/bash
# Pre-registered CIFAR-100 replication (paper/integration_carry/qa/PREREGISTRATION_CIFAR100_20261030.md).
cd "$(dirname "$0")"
export ACC_DATASET=cifar100
[ -f resnet20_cifar100.pt ] || python resnet20.py --epochs 50 --seed 0 > resnet20_cifar100_train.log
export ACC_THREADS=10
python cnn_accumulate.py --designs exact round_5 round_6 round_7 round_8 trunc_4 --n 10000 \
  --compensate calibrated --out cnn100_cal_10k.jsonl > cnn100_cal_10k_a.log 2>&1 &
python cnn_accumulate.py --designs trunc_6 trunc_7 evo_add8u_0E2 evo_add8u_6QU evo_add8u_2LL loa_5 --n 10000 \
  --compensate calibrated --out cnn100_cal_10k.jsonl > cnn100_cal_10k_b.log 2>&1 &
wait
python tost.py cnn100_cal_10k.jsonl
