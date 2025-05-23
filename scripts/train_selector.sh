export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=1

python ../src/train_selector.py \
    --use_deepspeed false \
    --model_name_or_path /d2/mxy/Models/Meta-Llama-3-8B \
    --train_data_path /d2/mxy/TASA/config/data_config_exp_12.json \
    --output_dir ../selector/selector_exp12_llama3-8B \
    --peft_type lora \
    --n_clusters 500 \
    --encoder embedding \
    --embedding_path /d2/mxy/TASA/models/Meta-Llama-3-8B/embedding.pth \
    --distance COS \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --do_train True \
    --use_instruction false \
    --use_output false \
    --bf16 true \
    --fp16 false \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --save_strategy "epoch" \
    --save_total_limit 5 \
    --learning_rate 4e-4 \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 512 \