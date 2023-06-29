import os
import argparse
import sys
import torch

from util import Logger, nowdt, mkdir


# os.environ['CUDA_VISIBLE_DEVICES']  = '1,2,7,8'
# os.environ['CUDA_VISIBLE_DEVICES']  = '0,1,2,3,4,5,6,7'
# os.environ['CUDA_VISIBLE_DEVICES']  = '2,3'
# os.environ['CUDA_VISIBLE_DEVICES']  = '0,1'
# os.environ['CUDA_VISIBLE_DEVICES']  = '3,4,5,6'
# os.environ['CUDA_VISIBLE_DEVICES']  = '2,3,8,9'
# os.environ['CUDA_VISIBLE_DEVICES']  = '2,3,4,5,6,7,8,9'


label_list = ["0", "1"]
num_labels = len(label_list)

# arg

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--i", default=0, type=int, required=False, help="Predict data split i th")
parser.add_argument('--max_seq_length', default=512, type=int)
parser.add_argument('--train_batch_size', default= 64, type=int)
parser.add_argument('--valid_batch_size', default= 1024, type=int)
parser.add_argument('--test_batch_size', default= 1024, type=int)
parser.add_argument('--valid_step', default= 1000, type=int, help="valid step to save model")
parser.add_argument('--gradient_accumulation_steps', default= 8, type=int)
parser.add_argument('--num_train_epochs', default= 3, type=int)
parser.add_argument('--learning_rate', default=5e-3, type=float)
parser.add_argument('--warmup_proportion', default=0.1, type=float)
parser.add_argument('--evaluate_every', default=15, type=int)
parser.add_argument('--n_epochs', default=100, type=int)
parser.add_argument('--patient', default=10, type=int)
parser.add_argument('--len_train', default=700000, type=int)
parser.add_argument('--num_labels', type=int, default=2)
parser.add_argument('--gpu_nums', type=int, default=4)
parser.add_argument("--dataset", type=str, default="CN_NO_TREATMENT", help="dataset type.")
parser.add_argument("--k", type=int, default=600, help="top k")
parser.add_argument("--neg_num", type=int, default=10, help="negative numbers for train")
parser.add_argument("--csv_slices", type=int, default=1200, help="large data into slices per len")
parser.add_argument("--root_path", type=str, default="/home/huyaxuan/pycharm/commonsense/", help="Path to model checkpoint (by default train from scratch).")
parser.add_argument("--use_gpu", type=bool, default=True, help="Flag enabling gpu usage.")
parser.add_argument("--data_length", type=int, default=1200, help="the length of test")
parser.add_argument("--gpu_num", default='2,3,4,5', type=str, required=False, help="gpu num")
parser.add_argument("--begin", default=None, type=int, required=False, help="begin num")
parser.add_argument("--end", default=None, type=int, required=False, help="end num")
parser.add_argument("--do_lower_case", default="False", type=str, required=False, help="do_lower_case")


args = parser.parse_args()

args.output_dir = os.path.join(os.path.join(args.root_path, "save_uncased_notsep"), args.dataset)
args.dataset_path = os.path.join(os.path.join(args.root_path, "synth_data"), args.dataset)
print(nowdt())

os.environ['CUDA_VISIBLE_DEVICES']  = args.gpu_num