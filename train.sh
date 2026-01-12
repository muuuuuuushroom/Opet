#!/bin/bash
export DEFAULT_MASTER_PORT=10006
export CUDA_VISIBLE_DEVICES=6,7
export DEFAULT_NPROC_PER_NODE=2
export DEFAULT_CONFIG_FILE_PATH='configs_con/Others/soybeam.yaml' # 'configs_con/Crowd/SHA.yaml'
export OMP_NUM_THREADS=4 # Set the number of OpenMP threads
# 'configs_con/Others/soybeam.yaml'
# "configs_con/TGRS/Ship.yaml"

# nohup sh train.sh> output_nohup/RS/Ship/base_s5.log 2>&1 &
# nohup sh train.sh> output_nohup/RS/Car/base_s2.log 2>&1 &
# nohup sh train.sh> output_nohup/RS/People/base_s2.log 2>&1 &
# nohup sh train.sh> base_costpoint0.log 2>&1 &

NPROC_PER_NODE=${2:-$DEFAULT_NPROC_PER_NODE}
MASTER_PORT=${3:-$DEFAULT_MASTER_PORT}
CONFIG_FILE_PATH=${4:-$DEFAULT_CONFIG_FILE_PATH}

echo "CUDA: $CUDA_VISIBLE_DEVICES"
echo "nproc_per_node: $NPROC_PER_NODE"
echo "master_port: $MASTER_PORT"
echo "config path: $CONFIG_FILE_PATH"

# python -m torch.distributed.launch \
#     --nproc_per_node=$NPROC_PER_NODE \
#     --master_port=$MASTER_PORT \
#     --use_env main.py \
#     --cfg="$CONFIG_FILE_PATH" \
torchrun \
  --nproc_per_node=$NPROC_PER_NODE \
  --master_port=$MASTER_PORT \
  main.py \
  --cfg "$CONFIG_FILE_PATH"
    # --resume="/data/zlt/RemoteSensePET/outputs/RS/Ship/base_s2/best_checkpoint.pth"


# nohup sh train.sh> output_nohup/RS/Car/t_bs16_ctxt64_quarterprobloss.log 2>&1 &
# nohup sh train.sh> output_nohup/RS/People/t_bs16_ctxt64_quarterprobloss.log 2>&1 &

# running:
# 0//: Car:     
# 1//:      
# 2//: People:  
# 3//:   
# 4//: Ship:    
# 5//: Ship:    
# 6//: Ship:    
# 7//:    

# Note: People: t_bs16_ctxt64_attn  (stop) 

# TGRS_base = noen + attn
# Running programs: t/w.o._ctxt64, t_noen, t_attn, t_base
# all exps are under the context patch of [128, 64] except for extra info
