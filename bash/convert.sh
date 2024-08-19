parent_dir=../..

python ${parent_dir}/RISE/workflow_convert.py \
--filepath ${parent_dir}/RISE/log/Llama-2-7b-chat-hf_gsm8k/2_turns.json \
--output ${parent_dir}/RISE/dataset/finetune.json \
--model_path meta-llama/Llama-2-7b-chat-hf \
--env gsm8k \
--criteria all \
--remove_duplication \
--shuffle 