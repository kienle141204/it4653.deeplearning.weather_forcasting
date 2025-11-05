import numpy as np

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

# def MAPE(pred, true):
#     mask = true != 0
#     return np.mean(np.abs((pred[mask] - true[mask])/true[mask]))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)

    return mae, mse, rmse, mape