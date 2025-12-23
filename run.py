import argparse

import argparse
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecasting
import torch
import warnings
import random
import numpy as np

warnings.filterwarnings('ignore')

def main():
    seed = 2025
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    parser = argparse.ArgumentParser(description='Run Weather Forecasting Experiment')
    # basic config
    parser.add_argument('--grid_size', type=tuple, default=(16, 16), help='grid size of the data')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    # parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='ConvLSTM',
                        help='model name, options: [ConvLSTM]')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # # data loader
    # parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/data.csv', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='./data/data.csv', help='data csv file')
    parser.add_argument('--features', action='store_false' , default=True, help='')
    parser.add_argument('--target', type=str, nargs='+', default=['t2m'], help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--use_multi_heads', type=int, default=0, help='number of used feature heads')

    # forecasting task
    parser.add_argument('--his_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length') # no longer needed in inverted Transformers
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    # ConvLSTM parameters
    parser.add_argument('--input_channels', type=int, default=7, help='input dimension')
    parser.add_argument('--hidden_channels', type=int, nargs='+', default=[64, 128], help='hidden dimensions for ConvLSTM layers')
    parser.add_argument('--num_hidden', type=int, default=[64, 64, 64, 64])
    parser.add_argument('--kernel_size', type=int, default=5, help='kernel size for ConvLSTM layers')
    parser.add_argument('--num_layers', type=int, default=2, help='number of ConvLSTM layers')
    parser.add_argument('--bias', action='store_true', help='whether to use bias in ConvLSTM layers', default=False)
    parser.add_argument('--batch_first', action='store_true', help='whether batch is first dimension', default=True)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--layer_norm', type=int, default=1)

    # scheduled sampling
    parser.add_argument('--reverse_scheduled_sampling', type=int, default=0)
    parser.add_argument('--scheduled_sampling', type=int, default=0)
    parser.add_argument('--r_sampling_step_1', type=float, default=25000)
    parser.add_argument('--r_sampling_step_2', type=int, default=50000)
    parser.add_argument('--r_exp_alpha', type=int, default=5000)

    # SwinLSTM parameters
    # parser.add_argument('--input_channels', default=1, type=int, help='Number of input image channels')
    parser.add_argument('--input_img_size', default=16, type=int, help='Input image size')
    parser.add_argument('--patch_size', default=2, type=int, help='Patch size of input images')
    parser.add_argument('--embed_dim', default=128, type=int, help='Patch embedding dimension')
    parser.add_argument('--depths', default=[2], type=int, help='Depth of Swin Transformer layer for SwinLSTM-B')
    parser.add_argument('--depths_down', default=[2, 6], type=int, help='Downsample of SwinLSTM-D')
    parser.add_argument('--depths_up', default=[6, 2], type=int, help='Upsample of SwinLSTM-D')
    parser.add_argument('--heads_number', default=[4, 8], type=int,
                        help='Number of attention heads in different layers')
    parser.add_argument('--window_size', default=2, type=int, help='Window size of Swin Transformer layer')
    parser.add_argument('--drop_rate', default=0., type=float, help='Dropout rate')
    parser.add_argument('--attn_drop_rate', default=0., type=float, help='Attention dropout rate')
    parser.add_argument('--drop_path_rate', default=0.1, type=float, help='Stochastic depth rate')

    # GraphWaveNet parameters
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate for GraphWaveNet')
    parser.add_argument('--adj_method', type=str, default='distance', choices=['distance', 'knn'],
                        help='method to build adjacency matrix: distance or knn')
    parser.add_argument('--adj_threshold', type=float, default=1.5, 
                        help='distance threshold for adjacency matrix (distance method)')
    parser.add_argument('--adj_k', type=int, default=8,
                        help='number of nearest neighbors for adjacency matrix (knn method)')

    # optimization
    parser.add_argument('--num_workers', type=int, default=2, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--lr_patience', type=int, default=2)
    parser.add_argument('--early_stop_patience', type=int, default=5)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

   
    args = parser.parse_args()

    # exp = Exp_Long_Term_Forecasting(args)
    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_sl{}_pl{}_lr{}_ep{}_hd{}_ss{}_mh{}'.format(
                        args.model,
                        # args.data,
                        args.his_len,
                        # args.label_len,
                        args.pred_len,
                        args.learning_rate,
                        args.train_epochs,
                        args.hidden_channels,
                        args.scheduled_sampling,
                        args.use_multi_heads)
            if not args.features:
                setting += '_in_c{}_ft_{}'.format(
                    args.input_channels,
                    args.target
                )
            if args.model == 'ConvLSTM':
                setting += '_ks{}_nl{}'.format(
                    args.kernel_size,
                    args.num_layers
                )
            else:
                setting += '_ps_{}_depths_{}'.format(
                    args.patch_size,
                    args.depths
                )

            exp = Exp_Long_Term_Forecasting(args)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            # if args.do_predict:
            #     print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            #     exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_sl{}_pl{}_lr{}_ep{}'.format(
                        args.model,
                        # args.data,
                        args.his_len,
                        # args.label_len,
                        args.pred_len,
                        args.learning_rate,
                        args.train_epochs)
        if not args.features:
            setting += '_in_c{}_ft_{}'.format(
                args.input_channels,
                args.target
            )
        if args.model == 'ConvLSTM':
            setting += '_ks{}_nl{}'.format(
                args.kernel_size,
                args.num_layers
            )
        else:
            setting += ''

        exp = Exp_Long_Term_Forecasting(args)
        # print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        # exp.test(setting, test=1)
        # torch.cuda.empty_cache()
    
    # Here you would typically load your config and start the experiment
    # print(f"Running experiment with config: {args.config}")

if __name__ == "__main__":
    main()