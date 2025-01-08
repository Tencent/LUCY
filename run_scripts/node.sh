#!/bin/bash
# set -e
# set -x

export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_P2P_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=25200

SCRIPT=$1
INDEX=$2
MASTER_ADDR="10.206.100.11"
export DISTRIBUTED_ARGS="
    --nproc_per_node 8 \
    --nnodes 2 \
    --node_rank $INDEX \
    --master_addr $MASTER_ADDR \
    --master_port 9996
"

./$SCRIPT

