parent_dir=../..
# env=gsm8k
env=math
data_path=${parent_dir}/RISE/data/${env}/demo.jsonl
student_model=Llama-2-7b-chat-hf



python ${parent_dir}/RISE/generation.py \
    --data_path ${data_path} \
    --env ${env} \
    --log_dir ${parent_dir}/RISE/log/${model_name}_${env}/ \
    --max_turns 2 \
    --num_of_samples 1 5 \
    --models ${model_name} gpt-3.5-turbo