export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS='8'

model_path=$1
shift
model_name=$1
shift
# benchmark=$1
# shift

python eval_mteb/run_mteb.py \
  --model ${model_path} \
  --model_name ${model_name} \
  --precision fp16 \
  --model_kwargs "{\"max_length\": 8192, \"attn_type\": \"causal\", \"pooler_type\": \"last\", \"do_norm\": true, \"use_instruction\": true, \"instruction_template\": \"Instruct: {}\nQuery:\", \"instruction_dict_path\": \"eval_mteb/scripts/task_prompts.json\", \"attn_implementation\":\"flash_attention_2\"}" \
  --run_kwargs "{\"save_predictions\": \"true\"}" \
  --output_dir results/mteb/${model_name} \
  --batch_size 16 \
  --langs "eng" \
  --benchmark "BEIR" $@

  