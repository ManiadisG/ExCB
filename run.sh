#!/usr/bin/env bash

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Available GPUs: $NUM_GPUS"
#echo "Expected input args.:"
#echo "1: Visible GPUs (e.g. \"0,1\")"
#echo "2: Preset"
#echo "3: Run name"
#echo "4: Complex_arg"

WANDB_SERVICE_WAIT=300

JOB_COMMAND="$0 $@"
VISIBLE_GPUS=$1
PRESET=$2
RUN_NAME=$3
COMPLEX_ARG=$4
OUTPUT_DIR=$5

if [ -z "$PRESET" ]
then
    PRESET="None"
fi
if [ -z "$COMPLEX_ARG" ]
then
    COMPLEX_ARG="None"
fi
if [ -z "$RUN_NAME" ]
then
    RUN_NAME="None"
fi

OUTPUT_DIR="./experiments/"
CHECKPOINT_PATH=$OUTPUT_DIR$RUN_NAME"/checkpoint.pth"

echo "Selected GPUs: $VISIBLE_GPUS"

if [ -z "$VISIBLE_GPUS" ]
then
    GPU_COUNT=$NUM_GPUS
    echo "Running on all $GPU_COUNT available GPUs"
    VISIBLE_GPUS="0"
    for ((number=1;number<$GPU_COUNT;number++))
    do
        VISIBLE_GPUS="$VISIBLE_GPUS,$number"
    done
    export CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS
else
    export CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS
    IFS=','
    read -a strarr <<< "$VISIBLE_GPUS"
    GPU_COUNT=${#strarr[*]}
    echo "Running on $GPU_COUNT GPUs: $VISIBLE_GPUS"
fi

echo "Preset: $PRESET"
echo "Run name: $RUN_NAME"
echo "Complex_arg: $COMPLEX_ARG"
echo "Output path: $OUTPUT_DIR"
echo "Checkpoint path: $CHECKPOINT_PATH"

WANDB__SERVICE_WAIT=600 python -m torch.distributed.run \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:0 \
    --nnodes 1 \
    --nproc_per_node $GPU_COUNT \
    main.py \
        --preset=$PRESET \
        --run_name=$RUN_NAME \
        --bash_command="$JOB_COMMAND" \
        --output_dir=$OUTPUT_DIR \
        --complex_arg=$COMPLEX_ARG
