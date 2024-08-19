parent_dir=../..
model=demo
env=gsm8k
data_path=${parent_dir}/RISE/data/gsm8k/demo.jsonl

python ${parent_dir}/RISE/workflow_eval.py \
    --model ${model} \
    --data_path ${data_path} \
    --env ${env} \
    --log_dir ${parent_dir}/RISE/log/${model}_${env}_parallel/ \
    --max_turns 1 \
    --num_of_samples 5