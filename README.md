# Weather Forecasting with Deep Learning

Deep learning models for weather time-series forecasting.

## Models

- **ConvLSTM** - Convolutional LSTM for spatiotemporal prediction
- **PredRNN** - Predictive RNN with spatiotemporal memory
- **SwinLSTM** - Swin Transformer + LSTM hybrid

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Training

```bash
python run.py \
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
    --use_multi_heads 0
```

### 3. Testing

```bash
python run.py \
    --model ConvLSTM \
    --data_path './data/2324.csv' \
    --batch_size 32 \
    --input_channels 1 \
    --features \
    --target 't2m' \
    --learning_rate 1e-3 \
    --is_training 0 \
    --train_epochs 500 \
    --his_len 24 \
    --pred_len 24 \
    --kernel_size 3 \
    --hidden_channels 32 64 \
    --num_layers 2 \
    --lr_patience 5 \
    --early_stop_patience 15 \
    --use_multi_heads 0
```

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--model` | Model architecture (ConvLSTM, PredRNN, SwinLSTM) |
| `--data_path` | Path to CSV data file |
| `--target` | Target variable to predict (e.g., t2m) |
| `--is_training` | 1 = train, 0 = test |
| `--his_len` | Input sequence length (hours) |
| `--pred_len` | Prediction horizon (hours) |
| `--hidden_channels` | Hidden layer sizes |
| `--num_layers` | Number of LSTM layers |
| `--lr_patience` | Epochs before reducing learning rate |
| `--early_stop_patience` | Epochs before early stopping |

## Project Structure

```
├── models/          # Model architectures
├── data_provider/   # Data loading utilities
├── exp/             # Experiment configurations
├── layers/          # Custom layers
├── utils/           # Helper functions
├── scripts/         # Training scripts
├── checkpoints/     # Saved models
└── results/         # Output results
```

## Requirements

See `requirements.txt` for dependencies.
