import os
from pytorch_pretrained_bert.tokenization import BertTokenizer
from data_util import read_triplets, get_3part_dataloader
from parse import *

# def train(args):
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
# step1: read data and load data
root_path = "/data/huyaxuan/commonsense/synth_data/"
args.neg_num = 0
print("origin id:")
origin_path = os.path.join(root_path, "CN")
triplets = read_triplets(os.path.join(origin_path, "train.txt"), data_len=10)
print(triplets)
train_dataloader = get_3part_dataloader(triplets, tokenizer, args, data_type="train")

print("No split id:")
no_treatment_path = os.path.join(root_path, "CN_NO_TREATMENT")
triplets = read_triplets(os.path.join(no_treatment_path, "train.txt"), data_len=10)
print(triplets)
train_dataloader = get_3part_dataloader(triplets, tokenizer, args, data_type="train")

print("UNKNOW id:")
unused_path = os.path.join(root_path, "CN_UNUSED")
triplets = read_triplets(os.path.join(unused_path, "train.txt"), data_len=10)
print(triplets)
train_dataloader = get_3part_dataloader(triplets, tokenizer, args, data_type="train")
