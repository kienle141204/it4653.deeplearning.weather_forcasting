source venv/Scripts/activate

!python run.py \
    --model ConvLSTM \
    --data_path './data/2324.csv' \
    --batch_size 32 \
    --input_channels 1 \
    --features \
    --target 't2m' \
    --learning_rate 1e-3 \
    --is_training 1 \
    --train_epochs 500 \
    --his_len 24 \
    --pred_len 24 \
    --kernel_size 3 \
    --hidden_channels 32 64 \
    --num_layers 2 \
    --lr_patience 5 \
    --early_stop_patience 15 \
    --use_multi_heads 0 \
    --scheduled_sampling 1