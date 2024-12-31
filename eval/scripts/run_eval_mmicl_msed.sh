#!/bin/bash
WORKSPACE=../
cd $WORKSPACE
nnodes=1
nproc=4
MASTER_ADDR=localhost
MASTER_PORT=5009
RANK=0
config_path=./configs/multitask_mmicl.yaml
out=./results/mmicl_msed
mkdir -p $out
python -m torch.distributed.launch \
    --nproc_per_node $nproc \
    --nnodes $nnodes \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --node_rank $RANK \
    main.py \
    --config_path $config_path \
    --out $out