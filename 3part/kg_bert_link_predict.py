from transformers import BertTokenizer, BertModel
import numpy as np
import torch
from torch.nn import CrossEntropyLoss, MSELoss
import logging
logger = logging.getLogger(__name__)

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import random
import os
import argparse

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn import metrics

import torch.nn as nn
import multiprocessing
import pickle

max_seq_length = 512
# os.environ['CUDA_VISIBLE_DEVICES']= '0'
# os.environ['CUDA_VISIBLE_DEVICES']= '0,1'
# os.environ['CUDA_VISIBLE_DEVICES']  = '4,5,6,7'
# os.environ['CUDA_VISIBLE_DEVICES']  = '1,3,5,7'
os.environ['CUDA_VISIBLE_DEVICES']  = '0,1,2,3,4,5,6,7'


label_list = ["0", "1"]
num_labels = len(label_list)

# arg
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_batch_size = 128
eval_batch_size = 128
test_batch_size = 128
predict_batch_size = 128
gradient_accumulation_steps = 1
num_train_epochs = 3
learning_rate = 5e-5
warmup_proportion = 0.1
# output_dir = "./output/64epoch5"
# output_dir = "./output/64-3"
# output_dir = "./output/" + str(train_batch_size)
do_train = True
do_predict = False
evaluate_every = 15
n_epochs = 100
patient = 10


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def convert_examples_to_features(examples, label_list, tokenizer):

    label_map = {label : i for i, label in enumerate(label_list)}
    # print(label_map)
    
    # features = []
    input_ids = []
    input_masks = []
    segment_ids = []
    label_ids = []
    for (ex_index, example) in enumerate(examples):
        # feature = []
        # print("example: ", example)
        if example[4] == "tail":
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
            tokens = ["[CLS]"] + tokens_a + tokens_b + ["[SEP]"]
            segment_id = [0] * len(tokens)

            # if tokens_b:
            #     tokens += tokens_b + ["[SEP]"]
            #     segment_id += [1] * (len(tokens_b) + 1)
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

            input_ids.append(input_id)
            input_masks.append(input_mask)
            segment_ids.append(segment_id)
            label_ids.append(label_id)

        elif example[4] =='head':
            # feature = []
            # print("example: ", example)
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

            # if tokens_b:
            #     tokens += tokens_b + ["[SEP]"]
            #     segment_id += [1] * (len(tokens_b) + 1)
            if tokens_b and tokens_c:
                tokens += tokens_b + tokens_c + ["[SEP]"]
                segment_id += [1] * (len(tokens_b) + len(tokens_c) + 1)        

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

            input_ids.append(input_id)
            input_masks.append(input_mask)
            segment_ids.append(segment_id)
            label_ids.append(label_id)

    return input_ids, input_masks, segment_ids, label_ids


def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def test_create_examples(set_type, lines, type):
    # print("count: ",count, "set_type: ", set_type, "-----------1------------")
    examples = []
    for (i, line) in tqdm(enumerate(lines)):
        head_ent_text = line[0]
        tail_ent_text = line[2]
        relation_text = line[1]

        if set_type == "test":
            # guid = "%s-%s" % (set_type, i)
            line = []
            text_a = head_ent_text
            text_b = relation_text
            text_c = tail_ent_text 
            label = "1"
            line.append(text_a)
            line.append(text_b)
            line.append(text_c)
            line.append(label)
            line.append(type)
            examples.append(line)
    return examples

def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
		print ("---  new folder...  ---")
		print ("---  OK  ---")
	else:
		print ("---  There is this folder!  ---")

def create_entity_dicts(all_tuples):
    e1_to_multi_e2 = {}
    e2_to_multi_e1 = {}

    for tup in all_tuples:
        e1, rel, e2 = tup

        if (e1, rel) in e1_to_multi_e2:
            e1_to_multi_e2[(e1, rel)].append(e2)
        else:
            e1_to_multi_e2[(e1, rel)] = [e2]

        if (e2, rel) in e2_to_multi_e1:
            e2_to_multi_e1[(e2, rel)].append(e1)
        else:
            e2_to_multi_e1[(e2, rel)] = [e1]
 
    return e1_to_multi_e2, e2_to_multi_e1

def predict(test_triples, entity_list, all_triples_str_set, e1_to_multi_e2, e2_to_multi_e1, output_dir=None, valid_model=None, valid_tokenizer=None):
    # import pdb; pdb.set_trace()
    if output_dir != None:
        model = BertForSequenceClassification.from_pretrained(output_dir, num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained(output_dir, do_lower_case=True)
        model.to(device)
    elif valid_model != None and valid_tokenizer != None:
        test_triples = random.sample(test_triples, 100)
        tokenizer = valid_tokenizer
        model = valid_model
        model = model.to(device)

    ranks = []
    ranks_left = []
    ranks_right = []

    hits_left = []
    hits_right = []
    hits = []

    top_ten_hit_count = 0
    for i in range(10):
        hits_left.append([])
        hits_right.append([])
        hits.append([])

    for test_triple in tqdm(test_triples):
        head = test_triple[0]
        relation = test_triple[1]
        tail = test_triple[2]
        #print(test_triple, head, relation, tail)

        multi_head = e2_to_multi_e1[(tail, relation)]
        multi_tail = e1_to_multi_e2[(head, relation)]

        head_corrupt_list = [test_triple]
        for corrupt_ent in entity_list:
            if corrupt_ent != head and corrupt_ent not in multi_head:
                tmp_triple = [corrupt_ent, relation, tail]
                tmp_triple_str = '\t'.join(tmp_triple)
                if tmp_triple_str not in all_triples_str_set:
                    # may be slow
                    head_corrupt_list.append(tmp_triple)

        tmp_examples = test_create_examples("test", head_corrupt_list, type='head')
        input_ids, input_masks, segment_ids, label_ids = convert_examples_to_features(                
            tmp_examples, label_list, tokenizer)

        all_input_ids = torch.tensor(input_ids, dtype=torch.long)
        all_input_masks = torch.tensor(input_masks, dtype=torch.long)
        all_segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        all_label_ids = torch.tensor(label_ids, dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_masks, all_segment_ids, all_label_ids)
        # Run prediction for temp data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=predict_batch_size, drop_last=True)
        
        model.eval()

        preds = []
        
        for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            
            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, labels=None)
            if len(preds) == 0:
                batch_logits = logits.detach().cpu().numpy()
                preds.append(batch_logits)

            else:
                batch_logits = logits.detach().cpu().numpy()
                preds[0] = np.append(preds[0], batch_logits, axis=0)       

        preds = preds[0]
        # get the dimension corresponding to current label 1


        rel_values = preds[:, all_label_ids[0]]
        rel_values = torch.tensor(rel_values)


        _, argsort1 = torch.sort(rel_values, descending=True)

        argsort1 = argsort1.cpu().numpy()

        top_list = []
        # top = []
        for i in range(10):
            top_rank = np.where(argsort1 == i)[0][0]
            top_list.append(head_corrupt_list[top_rank])
     
        # write top10 into file
        # with open("top10.txt","a") as f:
        #     f.write('--------------------------\n')
        #     f.write('head_corrupt_list top 10:\n')
        #     f.write(head + '\t' + relation + '\t' + tail +'\n')
        # with open("top10.txt","a") as f:
        #     for i in range(10):
        #         f.write(top_list[i][0] + '\t' + top_list[i][1] + '\t' + top_list[i][2] + "\n") 

        rank1 = np.where(argsort1 == 0)[0][0]
        ranks.append(rank1+1)
        ranks_left.append(rank1+1)
        if rank1 < 10:
            top_ten_hit_count += 1

        tail_corrupt_list = [test_triple]
        for corrupt_ent in entity_list:
            if corrupt_ent != tail and corrupt_ent not in multi_tail:
                tmp_triple = [head, relation, corrupt_ent]
                tmp_triple_str = '\t'.join(tmp_triple)
                if tmp_triple_str not in all_triples_str_set:
                    # may be slow
                    tail_corrupt_list.append(tmp_triple)

        tmp_examples = test_create_examples("test", tail_corrupt_list, type='tail')
        #print(len(tmp_examples))
        input_ids, input_masks, segment_ids, label_ids = convert_examples_to_features(                
            tmp_examples, label_list, tokenizer)

        all_input_ids = torch.tensor(input_ids, dtype=torch.long)
        all_input_masks = torch.tensor(input_masks, dtype=torch.long)
        all_segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        all_label_ids = torch.tensor(label_ids, dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_masks, all_segment_ids, all_label_ids)
        # Run prediction for temp data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=predict_batch_size, drop_last=True)
        model.eval()
        preds = []        

        for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            
            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, labels=None)
            if len(preds) == 0:
                batch_logits = logits.detach().cpu().numpy()
                preds.append(batch_logits)

            else:
                batch_logits = logits.detach().cpu().numpy()
                preds[0] = np.append(preds[0], batch_logits, axis=0) 

        preds = preds[0]
        # get the dimension corresponding to current label 1
        rel_values = preds[:, all_label_ids[0]]
        rel_values = torch.tensor(rel_values)
        _, argsort1 = torch.sort(rel_values, descending=True)

        argsort1 = argsort1.cpu().numpy()

        top_list = []
        # top = []
        for i in range(10):
            top_rank = np.where(argsort1 == i)[0][0]
            top_list.append(tail_corrupt_list[top_rank])
            
        # write top10 into file
        # with open("top10.txt","a") as f:
        #     f.write('--------------------------\n')
        #     f.write('tail_corrupt_list top 10:\n')
        #     f.write(head + '\t' + relation + '\t' + tail + '\n')
        # with open("top10.txt","a") as f:
        #     for i in range(10):
        #         f.write(top_list[i][0] + '\t' + top_list[i][1] + '\t' + top_list[i][2] + "\n") 

        rank2 = np.where(argsort1 == 0)[0][0]
        ranks.append(rank2+1)
        ranks_right.append(rank2+1)
        print('right: ', rank2)
        print('mean rank until now: ', np.mean(ranks))
        if rank2 < 10:
            top_ten_hit_count += 1
        print("hit@10 until now: ", top_ten_hit_count * 1.0 / len(ranks))
        print('----------',id,'-----------: ', 'Ranks ready!')

        for hits_level in range(10):
            if rank1 <= hits_level:
                hits[hits_level].append(1.0)
                hits_left[hits_level].append(1.0)
            else:
                hits[hits_level].append(0.0)
                hits_left[hits_level].append(0.0)

            if rank2 <= hits_level:
                hits[hits_level].append(1.0)
                hits_right[hits_level].append(1.0)
            else:
                hits[hits_level].append(0.0)
                hits_right[hits_level].append(0.0)

    if output_dir != None:
        path = output_dir
        mkdir(path+'/ranks/')
        mkdir(path+'/ranks_left/')
        mkdir(path+'/ranks_right/')
        mkdir(path+'/hits/')
        mkdir(path+'/hits_left/')
        mkdir(path+'/hits_right/')
        save_obj(ranks, path+'/ranks/'+str(id))
        save_obj(ranks_left, path+'/ranks_left/'+str(id))
        save_obj(ranks_right, path+'/ranks_right/'+str(id))
        save_obj(hits, path+'/hits/'+str(id))
        save_obj(hits_left, path+'/hits_left/'+str(id))
        save_obj(hits_right, path+'/hits_right/'+str(id))
        print('----------',id,'-----------: ', 'Saved ready!')
        
    return np.mean(1./np.array(ranks))



def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--i",
                        default=0,
                        type=int,
                        required=False,
                        help="Predict data split i th")
    parser.add_argument("--output_dir",
                        default="./output/" + str(train_batch_size),
                        # default="./output/" + '64epoch5',
                        type=str,
                        required=False,
                        help="model path")
    args = parser.parse_args()
    output_dir = args.output_dir
    mkdir(output_dir)

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=True)
    # prepare model
    # train_examples = None
    num_train_optimization_steps = 0

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    train_triples = []

    # --------------------load_data--------------------
    with open("train_triple.txt") as f:
        data = f.readlines()
        for inst in data:
            line = []
            inst = inst.strip()
            inst = inst.split('\t')
            src, rel, tgt = inst
            line.append(src)
            line.append(rel)
            line.append(tgt)
            train_triples.append(line)

    valid_triples = []
    valid_entity_list = []
    with open("valid_triple.txt") as f:
        data = f.readlines()
        for inst in data:
            line = []
            inst = inst.strip()
            inst = inst.split('\t')
            src, rel, tgt = inst
            line.append(src)
            line.append(rel)
            line.append(tgt)
            valid_entity_list.append(src)
            valid_entity_list.append(tgt)
            valid_triples.append(line)
    valid_entity_list = list(set(valid_entity_list))

    test_triples = []
    test_entity_list = []
    with open("test_triple.txt") as f:
        data = f.readlines()
        for inst in data:
            line = []
            inst = inst.strip()
            inst = inst.split('\t')
            src, rel, tgt = inst
            line.append(src)
            line.append(rel)
            line.append(tgt)
            test_entity_list.append(src)
            test_entity_list.append(tgt)
            test_triples.append(line)
    test_entity_list = list(set(test_entity_list))

    all_triples = train_triples + valid_triples + test_triples

    all_triples_str_set = set()
    for triple in all_triples:
        triple_str = '\t'.join(triple)
        all_triples_str_set.add(triple_str)

    e1_to_multi_e2, e2_to_multi_e1 = create_entity_dicts(all_triples)

    # --------------------train--------------------
    if do_train:
        train_head_examples = []
        with open("train_head_label.txt") as f:
            data = f.readlines()
            for inst in data:
                line = []
                inst = inst.strip()
                inst = inst.split('\t')
                # print(inst)
                # exit()
                src, rel, tgt, label = inst
                line.append(src)
                line.append(rel)
                line.append(tgt)
                line.append(label)
                line.append('head')
                train_head_examples.append(line)
        train_tail_examples = []
        with open("train_head_label.txt") as f:
            data = f.readlines()
            for inst in data:
                line = []
                inst = inst.strip()
                inst = inst.split('\t')
                # print(inst)
                # exit()
                src, rel, tgt, label = inst
                line.append(src)
                line.append(rel)
                line.append(tgt)
                line.append(label)
                line.append('tail')
                train_tail_examples.append(line)
        train_examples = train_head_examples
        train_examples.extend(train_tail_examples)
        # use a miniso to test
        # train_examples = train_examples[0:1000]
        print("Train data read finish!")
        input_ids, input_masks, segment_ids, label_ids = convert_examples_to_features(                
                train_examples, label_list, tokenizer)

        print("Get feature finish!")
        model = BertForSequenceClassification.from_pretrained("bert-base-cased",
                num_labels=2)

        if torch.cuda.device_count() > 1:  # multi-gpu
            print('Lets use', torch.cuda.device_count(), 'GPUs!')
            model = nn.DataParallel(model)
            # model = DDP(model)
        model.to(device)
        # model.to(device)
        
        num_train_optimization_steps = int(
            len(train_examples) / train_batch_size / gradient_accumulation_steps) * num_train_epochs
        print('number of batch per epoch:', num_train_optimization_steps)
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                            lr=learning_rate,
                            warmup=warmup_proportion,
                            t_total=num_train_optimization_steps)
        all_input_ids = torch.tensor(input_ids, dtype=torch.long)
        all_input_masks = torch.tensor(input_masks, dtype=torch.long)
        all_segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        all_label_ids = torch.tensor(label_ids, dtype=torch.long)

        train_data = TensorDataset(all_input_ids, 
                                    all_input_masks, 
                                    all_segment_ids, 
                                    all_label_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)
        print("Train data load finish!")
        # exit()
        best_valid_mrr = 0.
        model.train()
        for num_train_epoch in trange(int(num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                # define a new function to compute loss values for both output_modes
                logits = model(input_ids, segment_ids, input_mask, labels=None)
                #print(logits, logits.shape)
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                loss = loss / gradient_accumulation_steps
                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                print("Training loss: ", tr_loss, nb_tr_examples)
                # validation
                if (step + 1) % 1000: # TODO: number has to be checked
                    print('--- VALIDATION ---')
                    
                    valid_mrr = predict(valid_triples, valid_entity_list, all_triples_str_set, e1_to_multi_e2, e2_to_multi_e1, valid_model=model, valid_tokenizer = tokenizer)
                    # valid_mrr = predict(test_triples, test_entity_list, all_triples_str_set, e1_to_multi_e2, e2_to_multi_e1, valid_model=model, valid_tokenizer = tokenizer)
                    if valid_mrr > best_valid_mrr:
                        best_valid_mrr = valid_mrr
                        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

                        # If we save using the predefined names, we can load using `from_pretrained`
                        
                        path = output_dir + 'epoch' + str(num_train_epoch)
                        mkdir(path)
                        output_model_file = os.path.join(path, WEIGHTS_NAME)
                        output_config_file = os.path.join(path, CONFIG_NAME)

                        torch.save(model_to_save.state_dict(), output_model_file)
                        model_to_save.config.to_json_file(output_config_file)
                        tokenizer.save_vocabulary(path)
                
    # --------------------prediction--------------------
    if do_predict:
        test_mrr = predict(test_triples, test_entity_list, all_triples_str_set, e1_to_multi_e2, e2_to_multi_e1, output_dir=output_dir)
        print('MRR:', test_mrr)
if __name__ == '__main__':
    try:
        main()
        print("OK!")
    except KeyboardInterrupt:
        print('Interrupted')
