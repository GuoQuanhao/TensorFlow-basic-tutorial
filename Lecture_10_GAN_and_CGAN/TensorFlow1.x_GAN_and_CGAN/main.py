#-*- coding: utf-8 -*-
import tensorflow as tf
import argparse

from GAN import GAN
from CGAN import CGAN

from utils import check_args

## 解析和配置
def parse_args():
    # 创建解释器对象ArgumentParser
    parser = argparse.ArgumentParser(description="Tensorflow implementation of GAN Variants")
    # 添加可选参数
    parser.add_argument('--gan_type', type=str, default='GAN', choices=['GAN', 'CGAN'],
                        help='The type of GAN', required=True)
    parser.add_argument('--dataset', type=str, default='fashion-mnist', 
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=20, 
                        help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='The size of batch')
    parser.add_argument('--z_dim', type=int, default=62, 
                        help='Dimension of noise vector')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')

    return check_args(parser.parse_args())


def main():
    args = parse_args()
    if args is None:
      exit()

    models = [GAN, CGAN]
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
       
        gan = None
        for model in models:
            if args.gan_type == model.model_name:
                gan = model(sess,
                            epoch=args.epoch,
                            batch_size=args.batch_size,
                            z_dim=args.z_dim,
                            dataset_name=args.dataset,
                            checkpoint_dir=args.checkpoint_dir,
                            result_dir=args.result_dir,
                            log_dir=args.log_dir)
        if gan is None:
            raise Exception("[!] There is no option for " + args.gan_type)
        
        # 构建模型
        gan.build_model()
        gan.train()
        print(" [*] Training finished!")

        #可视化
        gan.visualize_results(args.epoch-1)
        print(" [*] Testing finished!")

if __name__ == '__main__':
    main()
