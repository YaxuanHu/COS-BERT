import os
# from socket import AF_AAL5
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
import csv
from torch.utils import data as torch_data

import argparse
import numpy as np
from collections import Counter
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification

from data_util import get_3part_dataloader
from data_util import read_triplets
from util import mkdir, Logger, nowdt
from metric_util import get_3part_rank
from conve import get_raw2id, get_id2raw, create_large_entity_dicts, ConvE, FB15KDataset, gen_candidates, write_in_csd

def built_sparse_test_file(test_heads: list, test_relations: list, test_tails: list, count_dic: dict, k: int, args):
    '''
    ********input*******
    test_heads, test_relations, test_tails, count_dic
    ********output*******
    new_head_list
    new_tail_list
    '''
    
    # tranverse the document
    new_head_list = []
    new_tail_list = []
    for i in range(len(test_heads)):
        if count_dic[test_heads[i]] == k:
            new_head_list.append(test_heads[i] + '\t' + test_relations[i] + '\t' + test_tails[i] + '\n')
        if count_dic[test_tails[i]] == k:
            new_tail_list.append(test_heads[i] + '\t' + test_relations[i] + '\t' + test_tails[i] + '\n')
            
    # mkdir save_sparse_file_path
    save_sparse_file_path = os.path.join(os.path.join(args.basic_path,'synth_data'), args.dataset + '_sparse')
    mkdir(save_sparse_file_path)
    
    # write into file
    with open(os.path.join(save_sparse_file_path,f'head_{k}.txt'),"w") as f:
        for i in range(len(new_head_list)):
            f.write(new_head_list[i])
    with open(os.path.join(save_sparse_file_path,f'tail_{k}.txt'),"w") as f:
        for i in range(len(new_tail_list)):
            f.write(new_tail_list[i])

def prefile(all_dict, model, i, head, tail):
    entity2id = all_dict['entity2id']
    relation2id = all_dict['relation2id']
    test_head_ids = all_dict['test_head_ids']
    test_tail_ids = all_dict['test_tail_ids']

    # step2: test dataLoader - - - - - -
    data_path = f'/data/huyaxuan/commonsense/synth_data/{args.dataset}_sparse/'
    if head:
        test_head_set = FB15KDataset(os.path.join(data_path, f'head_{i}.txt'), entity2id, relation2id)
        test_head_loader = torch_data.DataLoader(test_head_set, batch_size=args.conve_validation_batch_size)
    if tail:
        test_tail_set = FB15KDataset(os.path.join(data_path, f'tail_{i}.txt'), entity2id, relation2id)
        test_tail_loader = torch_data.DataLoader(test_tail_set, batch_size=args.conve_validation_batch_size)
    
    print("load test data success")
    
    # step3: generate labels and the top k candidates - - - - - -
    if head:
        head_candidatas = gen_candidates(test_head_ids, test_tail_ids, model, test_head_loader, len(relation2id), args, all_dict, 'head')
    if tail:
        tail_candidatas = gen_candidates(test_head_ids, test_tail_ids, model, test_tail_loader, len(relation2id), args, all_dict, 'tail')

    # step4: write into the file
    if head:
        write_in_csd(args, head_candidatas, 'head', i)
    if tail:
        write_in_csd(args, tail_candidatas, 'tail', i)

def read_csv(args, i, data_length=None):
    """
    datatype: head or tail
    """
    head_file = []
    
    data_path = '/data/huyaxuan/commonsense/output/'
    if os.path.exists(os.path.join(data_path, f'conve_{args.dataset}_{args.k}_head_{i}_candidates.csv')):
        path = os.path.join(data_path, f'conve_{args.dataset}_{args.k}_head_{i}_candidates.csv')
        with open(path, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                head_file.append(row)
    tail_file = []
    if os.path.exists(os.path.join(data_path, f'conve_{args.dataset}_{args.k}_tail_{i}_candidates.csv')):
        path = os.path.join(data_path, f'conve_{args.dataset}_{args.k}_tail_{i}_candidates.csv')
        with open(path, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                tail_file.append(row)
    if data_length:
        head_file = head_file[:data_length]
        tail_file = tail_file[:data_length]
    return {
        "heads":head_file,
        "tails":tail_file
    }

def count_hit(ranks, k):
    hit = np.count_nonzero(ranks < k)
    return hit

def mrr_3part(ranks, data_length):
    return (np.sum(1/(ranks+1)))/data_length

def stat_3part_metric(ranks, data_length, ks=(1, 3, 10)):
    for k in ks:
        print(f'Hits@{k}: {count_hit(ranks, k)/data_length:.4f}')
    # print(f'MR: {mr(ranks):.4f}')
    print(f'MRR: {mrr_3part(ranks, data_length):.4f}')


def predict_with_sparse(model, tokenizer, args, i):
    csv_data = read_csv(args, i)
    dataloader = get_3part_dataloader(csv_data, tokenizer, args, data_type="test")
    
    preds = []
    
    for input_ids, input_mask, segment_ids, label_ids in tqdm(dataloader):
        input_ids = input_ids.cuda()
        input_mask = input_mask.cuda()
        segment_ids = segment_ids.cuda()
        label_ids = label_ids.cuda()
        
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)
        
        preds.append(logits.detach().cpu().numpy())
    # pdb.set_trace()
    preds = np.concatenate(preds)[:, 1].reshape(-1, args.k)

    ranks = get_3part_rank(preds)
    # ---------------------stat_metric-------------------------
    data_path = f'/data/huyaxuan/commonsense/synth_data/{args.dataset}_sparse/'
    data_length = len(open(os.path.join(data_path, f'head_{i}.txt')).readlines())+len(open(os.path.join(data_path, f'tail_{i}.txt')).readlines())
    stat_3part_metric(ranks, data_length)
        
    # return mrr(ranks)

def sparse_sort(args):
    '''
    - - - - - input - - - - -
    file_path: the file path
    - - - - - output - - - - -
    null
    '''
    
    # if need to generate basic file
    if args.prepare_write_txt == 1:
        print(f'Star to generate basic file.')
        
        # step1: read file
        file_path = os.path.join(args.basic_path, args.dataset_path)
        train_heads, train_relations, train_tails = read_triplets(os.path.join(file_path, "train.txt"))
        test_heads, test_relations, test_tails = read_triplets(os.path.join(file_path, "test.txt"))
        
        entity_list = train_heads + train_tails + test_heads + test_tails
        print(f'Read file success!')
        
        # step2: count entity occirence num and build dic
        count_dic = Counter(entity_list)
        
        # count test entities number
        count_test = Counter(test_heads + test_tails)
        count = 0
        for i in count_test:
            count = count + 1
        print(f'Total entities in test is {count}.')
        
        # step3: bulit new head and tail txt file
        for i in range(args.minimum_threshold):
            built_sparse_test_file(test_heads, test_relations, test_tails, count_dic, i+1, args)
        print(f'Generate basic file success!')
        
        # step4: prefile with transE model
        # data prepare for conve
        entity2id = get_raw2id(train_heads + test_heads + train_tails + test_tails)
        id2entity = get_id2raw(entity2id)
        relation2id = get_raw2id(train_relations + test_relations)
        id2relation = get_id2raw(relation2id)
        print("Built 2id dict success!")

        test_head_ids = np.unique([entity2id[ent] for ent in test_heads])
        test_tail_ids = np.unique([entity2id[ent] for ent in test_tails])
        
        train_triples = np.stack((train_heads, train_relations, train_tails),axis=1)
        test_triples = np.stack((test_heads, test_relations, test_tails),axis=1)
        e1_to_multi_e2, e2_to_multi_e1 = create_large_entity_dicts(np.concatenate((train_triples, test_triples), axis=0), entity2id, relation2id)
        print("Create multi dict!")
        
        all_dict = {}
        all_dict['entity2id'] = entity2id
        all_dict['relation2id'] = relation2id
        all_dict['id2entity'] = id2entity
        all_dict['id2relation'] = id2relation
        all_dict['e1_to_multi_e2'] = e1_to_multi_e2
        all_dict['e2_to_multi_e1'] = e2_to_multi_e1
        all_dict['test_head_ids'] = test_head_ids
        all_dict['test_tail_ids'] = test_tail_ids
        print("load dict success")
        
        # load conve model
        # device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        # write---------------------------------
        model_path = '/data/huyaxuan/transE/save/conve_CN-82K_1000_checkpoint.tar'
        # model_path = '/data/huyaxuan/transE/save/conve_CN_600_checkpoint.tar'
        model = ConvE(args=args, num_entities=len(entity2id), num_relations=2*len(relation2id))  # type: torch.nn.Module
        model = model.cuda()
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("load model success")
        
        for i in range(args.minimum_threshold):
            head = 0
            tail = 0
            data_path = f'/data/huyaxuan/commonsense/synth_data/{args.dataset}_sparse/'
            if os.path.getsize(os.path.join(data_path, f'head_{i+1}.txt')):
                head = 1
            if os.path.getsize(os.path.join(data_path, f'tail_{i+1}.txt')):
                tail = 1
            prefile(all_dict, model, i+1, head, tail)
            # else:
            #     continue
        print('conve prefile success!')
    
    
    else:
        # load the best model
        # write----------------------
        # output_dir = '/data/huyaxuan/commonsense/save/CN/epoch1step53999/'
        output_dir = '/data/huyaxuan/commonsense/save/CN-82K/epoch1step63999/'
        print(f'The best model path: {output_dir}')
        model = BertForSequenceClassification.from_pretrained(output_dir, num_labels=2)
        tokenizer = BertTokenizer.from_pretrained(output_dir, do_lower_case=True)
        
        if torch.cuda.device_count() > 1:  # multi-gpu
            print('Lets use', torch.cuda.device_count(), 'GPUs!')
            model = nn.DataParallel(model)
            # model = DDP(model)
        model.cuda()

        model.eval()
        
        # predict_with_sparse
        for i in range(args.minimum_threshold):
            
            data_path = '/data/huyaxuan/commonsense/output/'
            # if os.path.exists(os.path.join(data_path, f'conve_{args.dataset}_{args.k}_head_{i+1}_candidates.csv')):
            print(f'file name: conve_{args.dataset}_{args.k}_head_{i+1}_candidates.csv')
            # if args.no_split_head_tail:
                # aaa
            # else:
            predict_with_sparse(model, tokenizer, args, i+1)
            # else:
            #     continue
            
    

def main(args):
    # write log
    log_file_path = os.path.join(os.path.join(args.basic_path, "log"), f'sparse_sort.txt')
    sys.stdout = Logger(log_file_path)
    print(nowdt())
    
    # print prepare_write_txt_type
    prepare_write_txt_type = args.prepare_write_txt
    print(f'prepare_write_txt type is {prepare_write_txt_type}.')
    
    # sparse_sort
    sparse_sort(args)
    
if __name__ == '__main__':
    # parse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare_write_txt', type=bool, default=0, help='prepare_write_txt')
    parser.add_argument('--no_split_head_tail', type=bool, default=1, help='no_split_head_tail')
    parser.add_argument('--minimum_threshold', type=int, default=10, help='minimum_threshold')
    parser.add_argument('--basic_path', type=str, default='/data/huyaxuan/commonsense/', help='basic_path')
    parser.add_argument("--dataset", type=str, default="CN-82K", help="dataset type.")
    parser.add_argument("--k", type=int, default=1000, help="top k")
    
    parser.add_argument('--embedding-dim', type=int, default=200, help='The embedding dimension (1D). Default: 200')
    parser.add_argument('--embedding-shape1', type=int, default=20, help='The first dimension of the reshaped 2D embedding. The second dimension is infered. Default: 20')
    parser.add_argument('--hidden-drop', type=float, default=0.3, help='Dropout for the hidden layer. Default: 0.3.')
    parser.add_argument('--input-drop', type=float, default=0.2, help='Dropout for the input embeddings. Default: 0.2.')
    parser.add_argument('--feat-drop', type=float, default=0.2, help='Dropout for the convolutional features. Default: 0.2.')
    parser.add_argument('--lr-decay', type=float, default=0.995, help='Decay the learning rate by this factor every epoch. Default: 0.995')
    parser.add_argument('--loader-threads', type=int, default=4, help='How many loader threads to use for the batch loaders. Default: 4')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess the dataset. Needs to be executed only once. Default: 4')
    parser.add_argument('--resume', action='store_true', help='Resume a model.')
    parser.add_argument('--use-bias', action='store_true', help='Use a bias in the convolutional layer. Default: True')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing value to use. Default: 0.1')
    parser.add_argument('--hidden-size', type=int, default=9728, help='The size of the hidden layer. The required size changes with the size of the embeddings. Default: 9728 (embedding size 200).')

    parser.add_argument("--conve_validation_batch_size", type=int, default=256, help="Maximum batch size during model validation.")

    
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
    args.dataset_path = os.path.join('synth_data', args.dataset)
    
    # main
    main(args)
