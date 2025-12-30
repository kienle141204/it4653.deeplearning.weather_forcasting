import torch
import warnings
import random
import numpy as np

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecasting
from utils.params import get_args, get_setting

warnings.filterwarnings('ignore')


def main():
    seed = 2025
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    args = get_args()

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = get_setting(args)

            exp = Exp_Long_Term_Forecasting(args)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            torch.cuda.empty_cache()
    else:
        setting = get_setting(args)

        exp = Exp_Long_Term_Forecasting(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)


if __name__ == "__main__":
    main()