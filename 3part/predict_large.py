import torch
import numpy as np
import os
import torch.nn as nn
import sys
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss, MSELoss
from pytorch_pretrained_bert.tokenization import BertTokenizer
import pdb

from data_util import get_3part_dataloader, read_csv
from util import nowdt, Logger
from metric_util import get_3part_rank, stat_3part_metric, mrr
from parse import *

def predict(csv_data, args, output_dir=None):
    # import pdb; pdb.set_trace()
    log_file_path = os.path.join(os.path.join(args.root_path, "log"), f'3part_{args.dataset}_test_{args.k}_log.txt')
    sys.stdout = Logger(log_file_path)
    # ---------------------model prepare-------------------------
    model = BertForSequenceClassification.from_pretrained(output_dir, num_labels=args.num_labels)
    tokenizer = BertTokenizer.from_pretrained(output_dir)
    # tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=args.do_lower_case)
    if torch.cuda.device_count() > 1:  # multi-gpu
        print('Lets use', torch.cuda.device_count(), 'GPUs!')
        model = nn.DataParallel(model)
        # model = DDP(model)
    model.cuda()

    model.eval()
    

    preds = []
    # ---------------------data slices-------------------------
    head_length = len(csv_data["heads"])
    tail_length = len(csv_data["tails"])
    csv_data_length = max(head_length, tail_length)
    times = csv_data_length // args.csv_slices
    for i in range(times + 1):
        begin = i * args.csv_slices
        end = begin + args.csv_slices
        head_end = end
        tail_end = end
        if head_end > csv_data_length: head_end = csv_data_length
        if tail_end > csv_data_length: tail_end = csv_data_length
        # if end >= 12000: break
        print(f'begin:{begin}_end:{end}')
        tmp_data = {} 
        if head_end > begin:
            tmp_data["heads"] = csv_data["heads"][begin:head_end]
        if tail_end > begin:
            tmp_data["tails"] = csv_data["tails"][begin:tail_end]
        # ---------------------load data-------------------------
        dataloader = get_3part_dataloader(tmp_data, tokenizer, args, data_type="test")

        # ---------------------test-------------------------
        
        for input_ids, input_mask, segment_ids, label_ids in tqdm(dataloader):
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()
            label_ids = label_ids.cuda()
            
            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, labels=None)
            
            preds.append(logits.detach().cpu().numpy())

    preds = np.concatenate(preds)[:, 1].reshape(-1, args.k)

    ranks = get_3part_rank(preds)
    # ---------------------stat_metric-------------------------
    stat_3part_metric(ranks, args.data_length)
        
    return mrr(ranks)

def main():
    # test_path = []
    # epoch_list = ["0001"]
    # # step_list = [
    # #             11999,13999,15999,17999,19999,\
    # #             21999,23999,25999,27999,29999,\
    # #             31999,33999
    # #             ]
    # step_list = [
    #             "42999"
    #             ]
    # for _, epoch_num in enumerate(epoch_list):
    #     for  _, step_num in enumerate(step_list):
    #         test_path.append(os.path.join(os.path.join(args.root_path, "save"), f'{args.dataset}/epoch_{epoch_num}_step{step_num}/'))
    # test_path = []
    # epoch_list = [1]
    # step_list = [
    #             63999
    #             ]
    # for _, epoch_num in enumerate(epoch_list):
    #     for  _, step_num in enumerate(step_list):
    #         test_path.append(os.path.join(os.path.join(args.root_path, "save"), f'{args.dataset}/epoch{epoch_num}step{step_num}/'))

    # print(test_path)
    # print(test_path)

    log_file_path = os.path.join(os.path.join(args.root_path, "log"), f'3part_{args.dataset}_test_{args.k}_log.txt')
    sys.stdout = Logger(log_file_path)

    csv_data = read_csv(args)
    # test_path = [os.path.join(args.root_path, 'save/CN_NO_TREATMENT/epoch_0000_step09999/')] #["/home/huyaxuan/.config/pytorch/bert-base-cased/"]  #ATOMIC
    # test_path = ['/data/huyaxuan/commonsense/save/ATOMIC/epoch_0001_step339999/'] #ATOMIC
    test_path = ['/data/huyaxuan/commonsense/save_uncased_notsep/ATOMIC_DEL/epoch_0001_step31999/'] 

    for value in range(len(test_path)):
        # output_dir = './output/64epoch0step9999/'
        output_dir = test_path[value]
        print(nowdt())
        print(output_dir)
        valid_mrr = predict(csv_data, args, output_dir=output_dir)
        print('MRR:', valid_mrr)

if __name__ == '__main__':
    main()

