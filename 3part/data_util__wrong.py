from torch import linspace
from tqdm import tqdm
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import numpy as np
import csv
import random
from tqdm import tqdm

from util import load_file
from parse import *

def read_random_head_tail_file(data_type, random = None, len_file = None):
    if random:
        head_file_name = data_type + "_random_head.txt"
        tail_file_name = data_type + "_random_tail.txt"
    else:
        head_file_name = data_type + "_head.txt"
        tail_file_name = data_type + "_tail.txt"
    head = load_file(head_file_name, len_file)
    tail = load_file(tail_file_name, len_file)
    return {
        'head': head,
        'tail': tail
    }

def read_csv(args, begin=0, end=None):
    """
    datatype: head or tail
    """
    if end:
        head_file = []
        path = os.path.join(os.path.join(args.root_path, 'data_prefilter'),f'conve_{args.dataset}_{args.k}_head_candidates.csv')
        with open(path, newline='') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i >= begin and i < end:
                    head_file.append(row)
        tail_file = []
        path = os.path.join(os.path.join(args.root_path, 'data_prefilter'),f'conve_{args.dataset}_{args.k}_tail_candidates.csv')
        with open(path, newline='') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i >= begin and i < end:
                    head_file.append(row)
    else:
        head_file = []
        path = os.path.join(os.path.join(args.root_path, 'data_prefilter'),f'conve_{args.dataset}_{args.k}_head_candidates.csv')
        with open(path, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                head_file.append(row)
        tail_file = []
        path = os.path.join(os.path.join(args.root_path, 'data_prefilter'),f'conve_{args.dataset}_{args.k}_tail_candidates.csv')
        with open(path, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                tail_file.append(row)
    return {
        "heads":head_file,
        "tails":tail_file
    }

def swapPositions(list, pos1, pos2): 
    list[pos1], list[pos2] = list[pos2], list[pos1]
    return list

def read_triplets(file_path, data_len=None):
    heads, relations, tails = [], [], []
    with open(file_path, "r") as f:
        for line in f:
            if (len(line.strip().split('\t')) == 3):
                head, relation, tail = line.strip().split('\t')
                heads.append(head)
                relations.append(relation)
                tails.append(tail)
    if data_len:
        heads = heads[:data_len]
        relations = relations[:data_len]
        tails = tails[:data_len]
    return {
        "heads": heads, 
        "relations": relations, 
        "tails": tails
        }
   
class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

def convert_examples_to_features(head, tail, label_list, tokenizer, max_seq_length):
    
    label_map = {label : i for i, label in enumerate(label_list)}
    # print(label_map)
    
    features = []
    for (ex_index, example) in enumerate(tqdm(head, tail)):
        # feature = []
        # print("example: ", example)
        # if example[4] == "tail":
        tokens_a = tokenizer.tokenize(example[0])

        tokens_b = None
        tokens_c = None

        if example[1] and example[2]:
            tokens_b = tokenizer.tokenize(example[1])
            tokens_c = tokenizer.tokenize(example[2])
            # Modifies `tokens_a`, `tokens_b` and `tokens_c`in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP], [SEP] with "- 4"
            # _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            # _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_seq_length - 4)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_id = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_id += [1] * (len(tokens_b) + 1)
        if tokens_c:
            tokens += tokens_c + ["[SEP]"]
            segment_id += [1] * (len(tokens_c) + 1)        

        input_id = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_id)

        padding = [0] * (max_seq_length - len(input_id))
        input_id += padding
        input_mask += padding
        segment_id += padding

        assert len(input_id) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_id) == max_seq_length
        label_id = label_map[example[3]]

        features.append(
            InputFeatures(input_ids=input_id,
                            input_mask=input_mask,
                            segment_ids=segment_id,
                            label_id=label_id))

    return features

def get_feature(h, r, t, tokenizer, max_seq_length, label):
    label_map = {label : i for i, label in enumerate(label_list)}

    tokens_a = tokenizer.tokenize(h)
    # if r.startswith("["):
    #     tokens_b = list(r.split(" "))
    # else:
    tokens_b = tokenizer.tokenize(r)
    tokens_c = tokenizer.tokenize(t)

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_id = [0] * len(tokens)

    tokens += tokens_b + ["[SEP]"]
    segment_id += [1] * (len(tokens_b) + 1)

    tokens += tokens_c + ["[SEP]"]
    segment_id += [0] * (len(tokens_c) + 1)        

    input_id = tokenizer.convert_tokens_to_ids(tokens)
    # print(f'entity head:{h}_relation:{r}_tail:{t}_input_id:{input_id}')

    input_mask = [1] * len(input_id)

    padding = [0] * (max_seq_length - len(input_id))
    input_id += padding
    input_mask += padding
    segment_id += padding

    assert len(input_id) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_id) == max_seq_length
    label_id = label_map[label]

    return InputFeatures(input_ids=input_id,
                        input_mask=input_mask,
                        segment_ids=segment_id,
                        label_id=label_id)

def convert_3part_features(triplets, tokenizer, args):
    features = []
    max_seq_length = args.max_seq_length
    for i in tqdm(range(len(triplets["heads"]))):
        features.append(get_feature(triplets["heads"][i], triplets["relations"][i], triplets["tails"][i], tokenizer, max_seq_length, "1"))
        for j in range(args.neg_num):
            features.append(get_feature(random.choice(triplets["heads"]), triplets["relations"][i], triplets["tails"][i], tokenizer, max_seq_length, "0"))
    for i in tqdm(range(len(triplets["tails"]))):
        features.append(get_feature(triplets["heads"][i], triplets["relations"][i], triplets["tails"][i], tokenizer, max_seq_length, "1"))
        for j in range(args.neg_num):
            features.append(get_feature(triplets["heads"][i], triplets["relations"][i], random.choice(triplets["tails"]), tokenizer, max_seq_length, "0"))
    return features

def convert_3part_test_features(test_data, tokenizer, args):
    features = []
    max_seq_length = args.max_seq_length
    head_data = test_data["heads"]
    tail_data = test_data["tails"]
    for i in tqdm(range(len(head_data))):
        features.append(get_feature(head_data[i][0], head_data[i][1], head_data[i][2], tokenizer, max_seq_length, "1"))
        for j in range(3, len(head_data[i])):
            features.append(get_feature(head_data[i][j], head_data[i][1], head_data[i][2], tokenizer, max_seq_length, "1"))
    for i in tqdm(range(len(tail_data))):
        features.append(get_feature(tail_data[i][0], tail_data[i][1], tail_data[i][2], tokenizer, max_seq_length, "1"))
        for j in range(3, len(tail_data[i])):
            features.append(get_feature(tail_data[i][0], tail_data[i][1], tail_data[i][j], tokenizer, max_seq_length, "1"))
    return features

def get_dataloader(triplets, tokenizer, data_type, batch_size, max_seq_length):
    features = convert_examples_to_features(triplets, label_list, tokenizer, max_seq_length)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_masks = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    data = TensorDataset(all_input_ids, all_input_masks, all_segment_ids, all_label_ids)
    # Run prediction for temp data
    if data_type == 'train':
        sampler = RandomSampler(data)
    else:
        sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader

def get_3part_dataloader(triplets, tokenizer, args, data_type):
    if data_type == "train":
        features = convert_3part_features(triplets, tokenizer, args)
    elif data_type == "test":
        features = convert_3part_test_features(triplets, tokenizer, args)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_masks = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    data = TensorDataset(all_input_ids, all_input_masks, all_segment_ids, all_label_ids)
    if data_type == 'train':
        sampler = RandomSampler(data)
        batch_size=args.train_batch_size
    else:
        sampler = SequentialSampler(data)
        batch_size=args.test_batch_size
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader

def get_test_dataloader(triplets, tokenizer, args):
    features = convert_3part_features(triplets, tokenizer, args)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_masks = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    data = TensorDataset(all_input_ids, all_input_masks, all_segment_ids, all_label_ids)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=args.train_batch_size)
    return dataloader

def get_indices(list_np):
    values, counts = np.unique(list_np, return_counts=True)
    indices = np.cumsum(counts)
    indices = np.insert(indices, 0, 0)
    indices = [np.arange(indices[i], indices[i+1]) for i in range(0, len(indices)-1)]
    return indices

def get_indices_from_counts(counts):
    # values, counts = np.unique(list_np, return_counts=True)
    indices = np.cumsum(counts)
    indices = np.insert(indices, 0, 0)
    indices = [np.arange(indices[i], indices[i+1]) for i in range(0, len(indices)-1)]
    return indices

# for valid and test
def data_prepare(head, tail, tokenizer, data_type, begin_num=None):
    gpu_nums = args.gpu_nums

    len_head = len(head)
    len_head = (len_head // gpu_nums) * gpu_nums
    head = head[:len_head]
    head_np = np.array(head)
    head_examples = head_np[:, 0:5].tolist()
    head_group_labels = head_np[:, 5].astype(np.int)
    head_indices = get_indices(head_group_labels)

    len_tail = len(tail)
    len_tail = (len_tail // gpu_nums) * gpu_nums
    tail = tail[:len_tail]
    tail_np = np.array(tail)
    tail_examples = tail_np[:, 0:5].tolist()
    tail_group_labels = tail_np[:, 5].astype(np.int)
    tail_indices = get_indices(tail_group_labels)
    # from pdb import set_trace
    # set_trace()
    if data_type == 'valid':
    
        sample_step = 12
        head_indices = head_indices[begin_num::sample_step]
        head_examples = np.array(head_examples)
        head_examples = head_examples[np.concatenate(head_indices)]
        head_examples = list(head_examples)
        head_indices_counts = [len(i) for i in head_indices]    
        head_indices = get_indices_from_counts(head_indices_counts)

        # head_dataloader = head_examples[head_indices[0][0]:head_indices[-1][-1]]
        head_dataloader = get_dataloader(head_examples, tokenizer, 'valid', args.valid_batch_size, args.max_seq_length)
        print("Valid head data load finish!")

        # tail_indices = tail_indices[begin_num:end_num]
        # tail_dataloader = tail_examples[tail_indices[0][0]:tail_indices[-1][-1]]
        
        tail_indices = tail_indices[begin_num::sample_step]
        tail_examples = np.array(tail_examples)
        tail_examples = tail_examples[np.concatenate(tail_indices)]
        tail_examples = list(tail_examples)
        tail_indices_counts = [len(i) for i in tail_indices]
        tail_indices = get_indices_from_counts(tail_indices_counts)

        tail_dataloader = get_dataloader(tail_examples, tokenizer, 'valid', args.valid_batch_size, args.max_seq_length)
        print("Valid tail data load finish!")
    
    elif data_type == 'test':
        head_dataloader = get_dataloader(head_examples, tokenizer, 'test', args.test_batch_size, args.max_seq_length)
        print("Test head data load finish!")

        tail_dataloader = get_dataloader(tail_examples, tokenizer, 'test', args.test_batch_size, args.max_seq_length)
        print("Test tail data load finish!")

    return{
        'head_indices': head_indices,
        'head_dataloader': head_dataloader,
        'tail_indices': tail_indices,
        'tail_dataloader': tail_dataloader
    }
