#!/usr/bin/env sh



export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7;
export NCCL_IB_DISABLE=1;
export NCCL_IBEXT_DISABLE=1;
TASK="copa";
NUM_EXPERTS=64;
CHECKPOINT="/workdir/switch-base-${NUM_EXPERTS}";
DATASET_PATH="/workdir/datasets/superglue-copa";
METRIC_PATH="/workdir/metrics/super_glue"
GS_LOCATION="gs://xcloud-shared/ayazdan/switch-moe/results";
export WANDB_MODE=offline;
cd "/workdir/MC-SMoE" || exit;

#############################
# Create savedir structure
#############################
OUTPUT_DIR="results/${TASK}/vanilla/switch-${NUM_EXPERTS}"
DATE_TIME=`date +"%Y-%m-%d-%H:%M:%S"`;
OUTPUT_DIR="${OUTPUT_DIR}-${DATE_TIME}/${HOST}";
mkdir -p "${OUTPUT_DIR}";

accelerate launch --config_file static/finetune_multi_gpu_8.yaml \
  mcsmoe/finetune-switch-transformers.py \
  --per_device_train_batch_size=8 \
  --per_device_eval_batch_size=64 \
  --gradient_accumulation_steps=1 \
  --checkpoint="${CHECKPOINT}" \
  --dataset_path="${DATASET_PATH}" \
  --metric_path="${METRIC_PATH}" \
  --num_epochs=80 \
  --no_eval_until_epochs=1 \
  --save_each_epoch=False \
  --preprocessing_num_workers=8 \
  --num_experts="${NUM_EXPERTS}" \
  --task="${TASK}" \
  --learning_rate=3e-5 \
  --warmup_steps=16 \
  --output_dir="${OUTPUT_DIR}"  2>&1 | tee -a "${OUTPUT_DIR}/output.log" && echo "DONE" > "${OUTPUT_DIR}/done.txt"

# NOTE: You will need to run wandb sync after doing so
gsutil -m cp -r "results/*" "${GS_LOCATION}/results";
gsutil -m cp -r "wandb/*" "${GS_LOCATION}/wandb";
