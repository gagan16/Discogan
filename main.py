import argparse
from discogan import DiscoGAN
import tensorflow as tf
import os



def parse_args():
    desc = "Tensorflow implementation of DCGAN"
    parser = argparse.ArgumentParser(description=desc)
    #add path to dataset or add daatset in dataset folder
    parser.add_argument('--dataset', type=str, default='try', help='[mnist / cifar10 / custom_dataset]')

    parser.add_argument('--epoch', type=int, default=30, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=10, help='The size of batch per gpu')


    parser.add_argument('--print_freq', type=int, default=12, help='The number of image_print_freqy')

    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate for generator')


    parser.add_argument('--img_size', type=int, default=128, help='The size of image')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')


    parser.add_argument('--result_dir', type=str, default='result',
                        help='Directory name to save the result')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory name to save the models')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.log_dir)

    #--models_dir
    check_folder(args.model_dir)

    #--result_dir
    check_folder(args.result_dir)



    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        gan = DiscoGAN(sess, args)
        # build graph
        gan.train()
        print(" [*] Training finished!")

if __name__ == '__main__':
    main()