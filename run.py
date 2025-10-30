import argparse
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecasting
import torch

def main():
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
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length') # no longer needed in inverted Transformers
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    # ConvLSTM parameters
    parser.add_argument('--input_channels', type=int, default=7, help='input dimension')
    parser.add_argument('--hidden_channels', type=int, nargs='+', default=[64, 128], help='hidden dimensions for ConvLSTM layers')
    parser.add_argument('--kernel_size', type=int, nargs='+', default=5, help='kernel size for ConvLSTM layers')
    parser.add_argument('--num_layers', type=int, default=2, help='number of ConvLSTM layers')
    parser.add_argument('--bias', action='store_true', help='whether to use bias in ConvLSTM layers', default=False)
    parser.add_argument('--batch_first', action='store_true', help='whether batch is first dimension', default=True)

    # SwinLSTM parameters
    # parser.add_argument('--input_channels', default=1, type=int, help='Number of input image channels')
    parser.add_argument('--input_img_size', default=16, type=int, help='Input image size')
    parser.add_argument('--patch_size', default=2, type=int, help='Patch size of input images')
    parser.add_argument('--embed_dim', default=128, type=int, help='Patch embedding dimension')
    parser.add_argument('--depths', default=[12], type=int, help='Depth of Swin Transformer layer for SwinLSTM-B')
    parser.add_argument('--depths_down', default=[2, 6], type=int, help='Downsample of SwinLSTM-D')
    parser.add_argument('--depths_up', default=[6, 2], type=int, help='Upsample of SwinLSTM-D')
    parser.add_argument('--heads_number', default=[4, 8], type=int,
                        help='Number of attention heads in different layers')
    parser.add_argument('--window_size', default=4, type=int, help='Window size of Swin Transformer layer')
    parser.add_argument('--drop_rate', default=0., type=float, help='Dropout rate')
    parser.add_argument('--attn_drop_rate', default=0., type=float, help='Attention dropout rate')
    parser.add_argument('--drop_path_rate', default=0.1, type=float, help='Stochastic depth rate')

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
            setting = '{}_sl{}_ll{}_pl{}_{}'.format(
                        args.model,
                        # args.data,
                        args.seq_len,
                        args.label_len,
                        args.pred_len,
                        ii)

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
        setting = '{}_sl{}_ll{}_pl{}_{}'.format(
            args.model,
            # args.data,
            args.seq_len,
            args.label_len,
            args.pred_len,
            ii)

        exp = Exp_Long_Term_Forecasting(args)
        # print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        # exp.test(setting, test=1)
        # torch.cuda.empty_cache()
    
    # Here you would typically load your config and start the experiment
    # print(f"Running experiment with config: {args.config}")

if __name__ == "__main__":
    main()