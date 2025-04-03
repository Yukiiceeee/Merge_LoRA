export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=4

python ../src/train_adapter.py \
    --use_deepspeed false \
    --model_name_or_path /d2/mxy/Models/Meta-Llama-3-8B \
    --domain med \
    --task ie \
    --train_data_path /d2/mxy/TASA/data/data_adapters/med/ie/train.json \
    --output_dir /d2/mxy/TASA/models/adapters/med/ie \
    --peft_type lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --model_max_length 512 \
    --save_strategy "epoch" \
    --save_total_limit 5 \
    --learning_rate 4e-4 \
    --num_train_epochs 5 \
    --logging_steps 10 \
    --do_train True \