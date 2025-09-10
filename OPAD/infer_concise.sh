#! /bin/bash
ratio=3.0

t=0.5
dz=400
p_id=0
model_name=qwen
output_dir='outputs/hh_outputs'
output_dir_log='outputs/hh_outputs/log'
mkdir -p $output_dir
mkdir -p $output_dir_log

python \
  infer_concise.py \
  --conv_type llama2 \
  --pref_prompt "Your answer should be creative as much as possible." \
  --model_path /home/v-zhaowzhang/models/Qwen2.5-7B-Instruct \
  --principle_id $p_id \
  --temperature $t \
  --ratio $ratio \
  --data_size $dz \
  --output_data_file ${output_dir}/${model_name}_hh_data.json \
  --output_result_file ${output_dir}/${model_name}_creative.json \
  1> ${output_dir_log}/${model_name}_creative.txt 


python \
  infer_concise.py \
  --conv_type llama2 \
  --pref_prompt "Your answer should be concise as much as possible." \
  --model_path /home/v-zhaowzhang/models/Qwen2.5-7B-Instruct \
  --principle_id $p_id \
  --temperature $t \
  --ratio $ratio \
  --data_size $dz \
  --output_data_file ${output_dir}/${model_name}_hh_data.json \
  --output_result_file ${output_dir}/${model_name}_concise.json \
  1> ${output_dir_log}/${model_name}_concise.txt 


python \
  infer_concise.py \
  --conv_type llama2 \
  --pref_prompt "Your answer should be verbose as much as possible." \
  --model_path /home/v-zhaowzhang/models/Qwen2.5-7B-Instruct \
  --principle_id $p_id \
  --temperature $t \
  --ratio $ratio \
  --data_size $dz \
  --output_data_file ${output_dir}/${model_name}_hh_data.json \
  --output_result_file ${output_dir}/${model_name}_verbose.json \
  1> ${output_dir_log}/${model_name}_verbose.txt 

python \
  infer_concise.py \
  --conv_type llama2 \
  --pref_prompt "Your answer should be uplifting as much as possible." \
  --model_path /home/v-zhaowzhang/models/Qwen2.5-7B-Instruct \
  --principle_id $p_id \
  --temperature $t \
  --ratio $ratio \
  --data_size $dz \
  --output_data_file ${output_dir}/${model_name}_hh_data.json \
  --output_result_file ${output_dir}/${model_name}_uplifting.json \
  1> ${output_dir_log}/${model_name}_uplifting.txt 


model_name=llama31

python \
  infer_concise.py \
  --conv_type llama2 \
  --pref_prompt "Your answer should be creative as much as possible." \
  --model_path /home/v-zhaowzhang/models/Meta-Llama-3.1-8B-Instruct \
  --principle_id $p_id \
  --temperature $t \
  --ratio $ratio \
  --data_size $dz \
  --output_data_file ${output_dir}/${model_name}_hh_data.json \
  --output_result_file ${output_dir}/${model_name}_creative.json \
  1> ${output_dir_log}/${model_name}_creative.txt 


python \
  infer_concise.py \
  --conv_type llama2 \
  --pref_prompt "Your answer should be concise as much as possible." \
  --model_path /home/v-zhaowzhang/models/Meta-Llama-3.1-8B-Instruct \
  --principle_id $p_id \
  --temperature $t \
  --ratio $ratio \
  --data_size $dz \
  --output_data_file ${output_dir}/${model_name}_hh_data.json \
  --output_result_file ${output_dir}/${model_name}_concise.json \
  1> ${output_dir_log}/${model_name}_concise.txt 


python \
  infer_concise.py \
  --conv_type llama2 \
  --pref_prompt "Your answer should be verbose as much as possible." \
  --model_path /home/v-zhaowzhang/models/Meta-Llama-3.1-8B-Instruct \
  --principle_id $p_id \
  --temperature $t \
  --ratio $ratio \
  --data_size $dz \
  --output_data_file ${output_dir}/${model_name}_hh_data.json \
  --output_result_file ${output_dir}/${model_name}_verbose.json \
  1> ${output_dir_log}/${model_name}_verbose.txt 

python \
  infer_concise.py \
  --conv_type llama2 \
  --pref_prompt "Your answer should be uplifting as much as possible." \
  --model_path /home/v-zhaowzhang/models/Meta-Llama-3.1-8B-Instruct \
  --principle_id $p_id \
  --temperature $t \
  --ratio $ratio \
  --data_size $dz \
  --output_data_file ${output_dir}/${model_name}_hh_data.json \
  --output_result_file ${output_dir}/${model_name}_uplifting.json \
  1> ${output_dir_log}/${model_name}_uplifting.txt 