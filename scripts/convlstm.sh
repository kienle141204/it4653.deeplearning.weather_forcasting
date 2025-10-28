source venv/Scripts/activate

python run.py \
    --model ConvLSTM \
    --data_path data/data.csv \
    --epochs 10 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --seq_len 20 \
    --pred_len 20 \
    --hidden_channels 64 \
    --kernel_size 3 \
    --save_model_path models/convlstm_weather.pth