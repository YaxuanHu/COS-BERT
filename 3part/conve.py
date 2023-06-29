import torch
import torch
from collections import Counter
from torch.utils import data
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
import numpy as np
import os
import csv
import pickle
from typing import Dict


import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True

def get_raw2id(raws):
    """
    raws: list of strs
    """
    counter = Counter(raws)
    ret_map = dict()
    for idx, (raw, _) in enumerate(counter.most_common()):
        ret_map[raw] = idx#
    return ret_map

def get_id2raw(raw2id):
    id2raw = dict()
    for raw, id in raw2id.items():
        id2raw[id] = raw
    return id2raw

def create_large_entity_dicts(all_tuples, entity2id, relation2id):
    e1_to_multi_e2 = {}
    e2_to_multi_e1 = {}
    e1_to_multi_e2_list = {}
    e2_to_multi_e1_list = {}

    for tup in all_tuples:
        e1, rel, e2 = tup
        e1 = entity2id[e1]
        rel = relation2id[rel]
        e2 = entity2id[e2]

        if (e1, rel) in e1_to_multi_e2:
            e1_to_multi_e2[(e1, rel)].append(e2)
        else:
            e1_to_multi_e2[(e1, rel)] = [e2]

        if (e2, rel) in e1_to_multi_e2:
            e1_to_multi_e2[(e2, rel)].append(e1)
        else:
            e1_to_multi_e2[(e2, rel)] = [e1]

        if (e2, rel+len(relation2id)) in e2_to_multi_e1:
            e2_to_multi_e1[(e2, rel+len(relation2id))].append(e1)
        else:
            e2_to_multi_e1[(e2, rel+len(relation2id))] = [e1]

    return e1_to_multi_e2, e2_to_multi_e1


class ConvE(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(ConvE, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities+1, args.embedding_dim, padding_idx=num_entities)
        self.emb_rel = torch.nn.Embedding(num_relations+1, args.embedding_dim, padding_idx=num_relations)
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.hidden_drop = torch.nn.Dropout(args.hidden_drop)
        self.feature_map_drop = torch.nn.Dropout2d(args.feat_drop)

        self.loss = torch.nn.BCELoss()
        self.emb_dim1 = args.embedding_shape1
        self.emb_dim2 = args.embedding_dim // self.emb_dim1

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=args.use_bias)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(args.embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities+1)))
        self.fc = torch.nn.Linear(args.hidden_size,args.embedding_dim)
        print(num_entities, num_relations)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded= self.emb_e(e1).view(-1, 1, self.emb_dim1, self.emb_dim2)
        rel_embedded = self.emb_rel(rel).view(-1, 1, self.emb_dim1, self.emb_dim2)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x= self.inp_drop(stacked_inputs)
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)

        x = self.feature_map_drop(x)

        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.mm(x, self.emb_e.weight.transpose(1,0))

        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)

        return pred
    
def multi_label(dict, input, entities_num):
    label = torch.zeros(len(input), entities_num)
    for i, key in enumerate(input):
        label[i].scatter_(0, torch.tensor(dict[(key[0], key[1])]), 1)
    return label

def get_index_oneline(input_index, multi_hot, groud_truth):
    multi_hot_index = np.nonzero(multi_hot)[0].tolist()
    a = set(input_index)
    b = set(multi_hot_index)
    output_index = a - b | set([groud_truth.item()])
    return list(output_index)

def get_index(input_index, multi_hot, groud_truth):
    output=[]
    for i in range(len(multi_hot)):
        # output.append(get_index_oneline(input_index, multi_hot, groud_truth[i]))
        output.append(get_index_oneline(input_index, multi_hot[i], groud_truth[i]))
    return output

def get_top_ids(pred, one_label_index, topk):
    output = []
    for i in range(len(pred)):
        tmp_pred = pred[i].cpu().numpy()
        tmp_one_label = np.array(one_label_index[i])
        
        tmp_pred = tmp_pred[tmp_one_label]
        sorted_index = np.argsort(tmp_pred)
        sorted_index = sorted_index[len(sorted_index)-topk:]
        output.append(tmp_one_label[sorted_index])
    return output

def get_triplets_and_entities(triplets, topk_ids, id2entity, id2relation, datatype):
    out = []
    count_fail = 0

    for i in range(len(triplets)):
        line = []
        line.append(id2entity[triplets[i][0]])
        line.append(id2relation[triplets[i][1]])
        line.append(id2entity[triplets[i][2]])
        if datatype == "head":
            groudtruth = triplets[i][0]
        elif datatype == "tail":
            groudtruth = triplets[i][2]
        candidates = topk_ids[i].tolist()
        # delete groud_truth from candidate
        if groudtruth in candidates:
            candidates.remove(groudtruth)
            for j in range(len(candidates)):
                candidate_j = id2entity[candidates[j]]
                line.append(candidate_j)
            out.append(line)
        else:
            count_fail = count_fail + 1
    return out, count_fail

def gen_candidates(head_ids, tail_ids, model, data_loader, rerelation_len, args,
        all_dict, head_or_tail
        ):
    head_labels = []
    relation_labels = []
    tail_labels = []

    head_topk_ids = []
    tail_topk_ids = []

    triplets = []
    
    id2entity = all_dict['id2entity']
    id2relation = all_dict['id2relation']
    e1_to_multi_e2 = all_dict['e1_to_multi_e2']
    e2_to_multi_e1 = all_dict['e2_to_multi_e1']

    tail_id2cls = {id:cls for cls, id in enumerate(tail_ids)}
    head_id2cls = {id:cls for cls, id in enumerate(head_ids)}
    
    head_index = [key for key in head_ids]
    tail_index = [key for key in tail_ids]

    model.eval()
    with torch.no_grad():
        for head, relation, tail in data_loader:
            head, relation, tail = head.cuda(), relation.cuda(), tail.cuda()

            head_labels.append(head.detach().cpu().numpy())
            relation_labels.append(relation.detach().cpu().numpy())
            tail_labels.append(tail.detach().cpu().numpy())
            
            if head_or_tail == 'head':
                pred_head = model.forward(tail, relation + rerelation_len)
                head2multi_hot = torch.Tensor(multi_label(e2_to_multi_e1, torch.stack((tail, relation+len(id2relation)), dim=1).tolist(), len(id2entity)+1))
                head_one_label_index = get_index(head_index, head2multi_hot, head)
                head_topk_ids.extend(get_top_ids(pred_head, head_one_label_index, args.k))
            # - - - - - - - - - - - - - - - - - - - - - - -
            
            if head_or_tail == 'tail':
                pred_tail = model.forward(head, relation)
                tail2multi_hot = torch.Tensor(multi_label(e1_to_multi_e2, torch.stack((head, relation), dim=1).tolist(), len(id2entity)+1))
                tail_one_label_index = get_index(tail_index, tail2multi_hot, tail)
                tail_topk_ids.extend(get_top_ids(pred_tail, tail_one_label_index, args.k))

    # if head_labels and 
    triplets = np.stack((np.concatenate(head_labels), np.concatenate(relation_labels), np.concatenate(tail_labels)), axis=1)
    
    if head_or_tail == 'head':
        head_triples_and_entities, head_fail = get_triplets_and_entities(triplets, head_topk_ids, id2entity, id2relation, head_or_tail)
        return head_triples_and_entities, head_fail
    if head_or_tail == 'tail':
        tail_triples_and_entities, tail_fail = get_triplets_and_entities(triplets, tail_topk_ids, id2entity, id2relation, head_or_tail)
        return tail_triples_and_entities, tail_fail
        

def write_in_csd(args, data, datatype, i):
    with open(os.path.join("./output",f'conve_{args.dataset}_{args.k}_{datatype}_{i}_candidates.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(data[0])
    # with open(os.path.join("./output",f'transe_{args.dataset}_{args.k}_{datatype}_fail.csv'), 'w', newline='', encoding='utf-8') as f:
    with open(os.path.join("./output",f'conve_{args.dataset}_{args.k}_{datatype}_{i}_fail.pkl'), 'wb') as f:
        pickle.dump(data[1], f, pickle.HIGHEST_PROTOCOL)

Mapping = Dict[str, int]

class FB15KDataset(data.Dataset):
    """Dataset implementation for handling FB15K and FB15K-237."""

    def __init__(self, data_path: str, entity2id: Mapping, relation2id: Mapping):
        self.entity2id = entity2id
        self.relation2id = relation2id
        with open(data_path, "r") as f:
            # data in tuples (head, relation, tail)
            self.data = [line.strip().split("\t") for line in f]

    def __len__(self):
        """Denotes the total number of samples."""
        return len(self.data)

    def __getitem__(self, index):
        """Returns (head id, relation id, tail id)."""
        head, relation, tail = self.data[index]
        head_id = self._to_idx(head, self.entity2id)
        relation_id = self._to_idx(relation, self.relation2id)
        tail_id = self._to_idx(tail, self.entity2id)
        return head_id, relation_id, tail_id

    @staticmethod
    def _to_idx(key: str, mapping: Mapping) -> int:
        try:
            return mapping[key]
        except KeyError:
            return len(mapping)
        
