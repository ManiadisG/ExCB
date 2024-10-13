#!/usr/bin/env bash


NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Available GPUs: $NUM_GPUS"
#echo "Expected input args.:"
#echo "1: Visible GPUs (e.g. \"0,1\")"


JOB_COMMAND="$0 $@"
VISIBLE_GPUS=$1
RUN_NAME=$2
EVAL_PRESET=$3
CHECKPOINT_PATH=$4
UPDATE_RUN=$5

if [ -z "$RUN_NAME" ]
then
    RUN_NAME="None"
fi
if [ -z "$UPDATE_RUN" ]
then
    UPDATE_RUN="False"
fi
if [ -z "$EVAL_PRESET" ]
then
    EVAL_PRESET=$PRESET
fi
if [ -z "$CHECKPOINT_PATH" ]
then
    CHECKPOINT_PATH=$OUTPUT_DIR$RUN_NAME"/checkpoint.pth"
fi

MODEL_VERSION="teacher"
OUTPUT_DIR="./experiments/"
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

echo "Selected GPUs: $VISIBLE_GPUS"
echo "Run name: $RUN_NAME"
echo "Eval preset: $EVAL_PRESET"
echo "Checkpoint path: $CHECKPOINT_PATH"
echo "Model version: $MODEL_VERSION"
echo "Update Run: $UPDATE_RUN"

git fetch --all
git reset --hard origin/master

echo "Running linear evaluation"
python -m torch.distributed.run \
--rdzv_backend c10d \
--rdzv_endpoint localhost:0 \
--nnodes 1 \
--nproc_per_node $GPU_COUNT \
engine/linear_eval/main.py \
    --preset=$EVAL_PRESET \
    --run_name=$RUN_NAME \
    --bash_command="$JOB_COMMAND" \
    --output_dir=$OUTPUT_DIR \
    --model_version=$MODEL_VERSION \
    --update_run=$UPDATE_RUN \
    --path_to_model=$CHECKPOINT_PATH