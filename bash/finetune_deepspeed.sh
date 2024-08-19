parent_dir=../..
model_path=meta-llama/Llama-2-7b-chat-hf
data_path=${parent_dir}/RISE/dataset/demo.json
model_save_dir=/scratch/bcls/yqu1/demo
epoch=0.3

deepspeed --master_port 32079 ${parent_dir}/RISE/FastChat/fastchat/train/train_mem.py \
    --deepspeed ${parent_dir}/RISE/deepspeed/zero3.json \
    --model_name_or_path ${model_path} \
    --data_path ${data_path} \
    --bf16 True \
    --output_dir ${model_save_dir} \
    --num_train_epochs ${epoch} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "steps" \
    --eval_steps 100000 \
    --save_strategy "steps" \
    --save_steps 100000 \
    --save_total_limit 8 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess False \
    --report_to wandb