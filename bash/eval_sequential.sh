parent_dir=../..
model=demo
env=gsm8k
data_path=${parent_dir}/RISE/data/gsm8k/demo.jsonl
context_window=2

python ${parent_dir}/RISE/workflow_eval.py \
    --model ${model} \
    --data_path ${data_path} \
    --env ${env} \
    --log_dir ${parent_dir}/log/${model}_${env}_sequential/ \
    --max_turns 5 \
    --num_of_samples 1 1 1 1 1 \
    --context_window ${context_window} \