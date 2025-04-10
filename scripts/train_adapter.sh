export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=1

python ../src/train_adapter.py \
    --use_deepspeed false \
    --model_name_or_path /d2/mxy/Models/Qwen2-7B \
    --domain med \
    --task mc \
    --train_data_path /d2/mxy/TASA/data/data_adapters/med/mc/train.json \
    --output_dir /d2/mxy/TASA/models/adapters \
    --peft_type lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --model_max_length 1024 \
    --save_strategy "steps" \
    --save_steps 100 \
    --learning_rate 2e-4 \
    --num_train_epochs 6 \
    --warmup_steps 40 \
    --logging_steps 1 \
    --do_train True \
    --weight_decay 0.01 \
    # --evaluation_strategy "steps" \
    # --eval_steps 100 \