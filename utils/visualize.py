import matplotlib.pyplot as plt
import pandas as pd  
from datetime import datetime
import numpy as np

def time_features_to_label(time_feat):
    if time_feat is None or len(time_feat) == 0:
        return None

    time_feat = np.asarray(time_feat)
    months = time_feat[:, 0].astype(int)
    days   = time_feat[:, 1].astype(int)
    hours  = time_feat[:, 3].astype(int)

    labels = []
    for m, d, h in zip(months, days, hours):
        dt = datetime(2024, m, d, h, 0)
        label = dt.strftime("%m/%d\n%H:%M")
        labels.append(label)
    
    return np.array(labels)

def visualize(historical_data, true_data, predicted_data, 
              x_mark=None, y_mark=None, 
              title="True vs Predicted", 
              xlabel="Time", ylabel="Value", 
              save_path=None):
    
    plt.figure(figsize=(14, 6))

    x_mark = time_features_to_label(x_mark) if x_mark is not None else None
    y_mark = time_features_to_label(y_mark) if y_mark is not None else None
    
    if x_mark is not None:
        x_len = len(x_mark)
        if x_len == 6:
            step = 2
        elif x_len == 12:
            step = 3
        elif x_len == 24:
            step = 6
        else:
            step = max(1, x_len // 6)
    
    if y_mark is not None:
        y_len = len(y_mark)
        if y_len == 6:
            y_step = 2
        elif y_len == 12:
            y_step = 3
        elif y_len == 24:
            y_step = 6
        else:
            y_step = max(1, y_len // 6)
    
    if x_mark is not None and y_mark is not None:
        plt.plot(x_mark, historical_data, label='Historical Data', color='blue', linewidth=2)
        plt.plot(y_mark, true_data, label='True Future', color='green', linewidth=2)
        plt.plot(y_mark, predicted_data, label='Predicted', color='red', linestyle='--', linewidth=2)
        
        all_labels = list(x_mark) + list(y_mark)
        all_positions = list(range(len(all_labels)))
        selected_positions = list(range(0, len(x_mark), step)) + list(range(len(x_mark), len(all_labels), y_step))
        selected_labels = [all_labels[i] for i in selected_positions]
        plt.xticks(ticks=[all_labels[i] for i in selected_positions], labels=selected_labels, rotation=45, ha='right')
    else:
        total_length = len(historical_data) + len(true_data)
        plt.plot(range(len(historical_data)), historical_data, label='Historical Data', color='blue', linewidth=2)
        plt.plot(range(len(historical_data), total_length), true_data, label='True Future', color='green', linewidth=2)
        plt.plot(range(len(historical_data), total_length), predicted_data, label='Predicted', color='red', linestyle='--', linewidth=2)
    
    if x_mark is not None and y_mark is not None:
        if len(x_mark) > 0 and len(y_mark) > 0:
            last_historical_time = x_mark[-1]
            plt.axvline(x=last_historical_time, color='black', linestyle=':', linewidth=1, label='Forecast Start')
    
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def visualize_frame(historical_data, true_data, predicted_data,
                    x_mark=None, y_mark=None,
                    title="Spatio-Temporal Forecasting",
                    save_path=None):
    
    # Chuẩn hóa dữ liệu (T, H, W)
    def preprocess(frames):
        if frames.ndim == 4:      # (T, C, H, W)
            return frames[:, 0]
        elif frames.ndim == 3:    # (T, H, W)
            return frames
        else:
            raise ValueError(f"Unsupported shape: {frames.shape}")
    
    # Chuyển time features → chuỗi đẹp
    if x_mark is not None:
        x_mark = time_features_to_label(x_mark)
    if y_mark is not None:
        y_mark = time_features_to_label(y_mark)

    hist = preprocess(historical_data)
    true = preprocess(true_data)
    pred = preprocess(predicted_data)

    T_hist = hist.shape[0]
    T_pred = true.shape[0]
    cols   = max(T_hist, T_pred)

    # Tăng chiều cao để có chỗ cho tên hàng
    fig = plt.figure(figsize=(3.5 * cols + 2, 10))

    fig.text(0.06, 0.80, "Historical",   fontsize=16, fontweight='bold', color='navy',    rotation=90, va='center')
    fig.text(0.06, 0.53, "Ground Truth", fontsize=16, fontweight='bold', color='darkgreen', rotation=90, va='center')
    fig.text(0.06, 0.26, "Predicted",    fontsize=16, fontweight='bold', color='crimson',  rotation=90, va='center')

    for t in range(T_hist):
        ax = fig.add_subplot(3, cols, t + 1)
        im = ax.imshow(hist[t], cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
        label = x_mark[t] if x_mark is not None else f"t−{T_hist-t}"
        ax.set_title(label, fontsize=11, pad=10)
        ax.axis('off')

    # Điền chỗ trống
    for t in range(T_hist, cols):
        ax = fig.add_subplot(3, cols, t + 1)
        ax.axis('off')

    for t in range(T_pred):
        ax = fig.add_subplot(3, cols, cols + t + 1)
        im = ax.imshow(true[t], cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
        label = y_mark[t] if y_mark is not None else f"t+{t+1}"
        ax.set_title(label, fontsize=11, pad=10, color='green')
        ax.axis('off')

    for t in range(T_pred):
        ax = fig.add_subplot(3, cols, 2*cols + t + 1)
        im = ax.imshow(pred[t], cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
        label = y_mark[t] if y_mark is not None else "Pred"
        ax.set_title(label, fontsize=10, pad=10, color='red', fontweight='bold')
        ax.axis('off')

    plt.subplots_adjust(left=0.12, right=0.88, top=0.92, bottom=0.08, hspace=0.3, wspace=0.05)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Normalized Value', rotation=270, labelpad=20, fontsize=12)

    fig.suptitle(title, fontsize=20, fontweight='bold', y=0.98)

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
