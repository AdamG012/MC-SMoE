#!/usr/bin/env sh



export CUDA_VISIBLE_DEVICES=0,1,2,3;
export NCCL_IB_DISABLE=1;
export NCCL_IBEXT_DISABLE=1;
TASK="squad";
NUM_EXPERTS=32;
CHECKPOINT="/workdir/switch-base-${NUM_EXPERTS}";
DATASET_PATH="/workdir/datasets/glue-sst2";
METRIC_PATH="/workdir/metrics/glue"
GS_LOCATION="gs://xcloud-shared/ayazdan/switch-moe/results";
export WANDB_MODE=offline;
cd "/workdir/MC-SMoE/" || exit;

#############################
# Create savedir structure
#############################
OUTPUT_DIR="results/${TASK}/vanilla/switch-${NUM_EXPERTS}"
DATE_TIME=`date +"%Y-%m-%d-%H:%M:%S"`;
OUTPUT_DIR="${OUTPUT_DIR}-${DATE_TIME}/${HOST}";
mkdir -p "${OUTPUT_DIR}";

# STEP 1 RUN
accelerate launch --config_file static/finetune_multi_gpu_4.yaml \
  mcsmoe/finetune-switch-transformers.py \
  --per_device_train_batch_size=8 \
  --per_device_eval_batch_size=64 \
  --gradient_accumulation_steps=1 \
  --checkpoint="${CHECKPOINT}" \
  --dataset_path="${DATASET_PATH}" \
  --metric_path="${METRIC_PATH}" \
  --num_epochs=20 \
  --no_eval_until_epochs=1 \
  --save_each_epoch=False \
  --preprocessing_num_workers=8 \
  --num_experts="${NUM_EXPERTS}" \
  --task="${TASK}" \
  --learning_rate=3e-5 \
  --warmup_steps=16 \
  --output_dir="${OUTPUT_DIR}" 2>&1 | tee -a "${OUTPUT_DIR}/output.log" && echo "DONE" > "${OUTPUT_DIR}/done.txt"

# STEP 2 RUN
OUTPUT_PERM_DIR="results/${TASK}/permuted/switch-${NUM_EXPERTS}"
DATE_TIME=`date +"%Y-%m-%d-%H:%M:%S"`;
OUTPUT_PERM_DIR="${OUTPUT_PERM_DIR}-${DATE_TIME}/${HOST}";
# This gets auto-created by permute-model
# mkdir -p "${OUTPUT_PERM_DIR}";
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file static/finetune_config.yaml \
  mcsmoe/permute-model.py \
  --checkpoint="${OUTPUT_DIR}/latest" \
  --save_dir="${OUTPUT_PERM_DIR}" \
  --strategy="weight" \
  --include_wo=True

# STEP 3 RUN
OUTPUT_MERGE_DIR="results/${TASK}/msmoe/switch-${NUM_EXPERTS}"
DATE_TIME=`date +"%Y-%m-%d-%H:%M:%S"`;
OUTPUT_MERGE_DIR="${OUTPUT_MERGE_DIR}-${DATE_TIME}/${HOST}";
mkdir -p "${OUTPUT_MERGE_DIR}";
accelerate launch --config_file static/finetune_multi_gpu_4.yaml \
 mcsmoe/msmoe-merging.py \
  --per_device_train_batch_size=16 \
  --per_device_eval_batch_size=16 \
  --gradient_accumulation_steps=1 \
  --checkpoint="${CHECKPOINT}" \
  --dataset_path="${DATASET_PATH}" \
  --metric_path="${METRIC_PATH}" \
  --preprocessing_num_workers=8 \
  --num_epochs=10 \
  --no_eval_until_epochs=0 \
  --num_eval_steps=50 \
  --learning_rate=2e-4 \
  --warmup_steps=16 \
  --weight_decay=0.01 \
  --kd_temperature=2 \
  --mlm_lambda=1.0 \
  --kd_lambda=0.2 \
  --hd_lambda=0 \
  --task="${TASK}" \
  --merging_strategy="normal" \
  --exact_fisher=False \
  --num_samples_for_merging=256 \
  --similarity_base="router-logits" \
  --reverse_similarity=False \
  --num_groups=8 \
  --globally_group=True \
  --permute_when_merge=False \
  --save_stable_rank=False \
  --encoder_merging_layers="3,5,7,9,11" \
  --decoder_merging_layers="1,3,5,7,9,11" \
  --output_dir="${OUTPUT_MERGE_DIR}" \
  --student_checkpoint="${OUTPUT_PERM_DIR}" \
  --teacher_checkpoint="${OUTPUT_PERM_DIR}" 2>&1 | tee -a "${OUTPUT_MERGE_DIR}/output.log" && echo "DONE" > "${OUTPUT_MERGE_DIR}/done.txt";

# STEP 4 RUN
OUTPUT_COMP_DIR="results/${TASK}/mcsmoe/switch-${NUM_EXPERTS}"
DATE_TIME=`date +"%Y-%m-%d-%H:%M:%S"`;
OUTPUT_COMP_DIR="${OUTPUT_COMP_DIR}-${DATE_TIME}/${HOST}";
mkdir -p "${OUTPUT_COMP_DIR}";
accelerate launch --config_file static/finetune_multi_gpu_4.yaml \
  mcsmoe/losparse-downstream.py \
  --per_device_train_batch_size=8 \
  --per_device_eval_batch_size=128 \
  --gradient_accumulation_steps=1 \
  --checkpoint="${CHECKPOINT}" \
  --dataset_path="${DATASET_PATH}" \
  --metric_path="${METRIC_PATH}" \
  --preprocessing_num_workers=8 \
  --num_epochs=50 \
  --no_eval_until_epochs=0 \
  --num_eval_steps=100 \
  --learning_rate=3e-5 \
  --warmup_steps=50 \
  --weight_decay=0.01 \
  --kd_temperature=2 \
  --mlm_lambda=1.0 \
  --kd_lambda=0.2 \
  --hd_lambda=0.0 \
  --task="${TASK}" \
  --output_dir="${OUTPUT_COMP_DIR}" \
  --teacher_checkpoint="${OUTPUT_PERM_DIR}" \
  --student_checkpoint="${OUTPUT_MERGE_DIR}/normal/router-logits/latest" \
  --final_threshold=0.10 \
  --low_rank_factor=32 2>&1 | tee -a "${OUTPUT_COMP_DIR}/output.log" && echo "DONE" > "${OUTPUT_COMP_DIR}/done.txt";

# NOTE: You will need to run wandb sync after doing so
gsutil -m cp -r "results/*" "${GS_LOCATION}/results";
gsutil -m cp -r "wandb/*" "${GS_LOCATION}/wandb";
