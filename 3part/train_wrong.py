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


from data_util import read_triplets, get_3part_dataloader
from util import Logger, nowdt, mkdir
from parse import *

def train(args):
    # device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case = args.do_lower_case)
    # tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    num_train_optimization_steps = 0

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    # step1: read data and load data
    path = args.dataset_path
    triplets = read_triplets(os.path.join(path, "train.txt"))
    print("Train data read finish!")

    train_dataloader = get_3part_dataloader(triplets, tokenizer, args, data_type="train")
    print("Train data load finish!")

    # step2: load and init model
    # model = BertForSequenceClassification.from_pretrained("/home/huyaxuan/.config/pytorch/bert-base-uncased/",num_labels=2)
    model = BertForSequenceClassification.from_pretrained("bert-base-cased",num_labels=2)
    print("Load model!")

    if torch.cuda.device_count() > 1:  # multi-gpu
        print('Lets use', torch.cuda.device_count(), 'GPUs!')
        model = nn.DataParallel(model)
    model.cuda()
    
    num_train_optimization_steps = int(
        len(triplets["heads"]) * 2 *( 1 + args.neg_num ) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    print('number of batch per epoch:', num_train_optimization_steps)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.00},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.00}
        ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                        lr=args.learning_rate,
                        warmup=args.warmup_proportion,
                        t_total=num_train_optimization_steps)
    loss_fct = CrossEntropyLoss()

    # step3: train
    model.train()
    for num_train_epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        # tr_loss = 0
        tr_loss = []
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.cuda() for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            # define a new function to compute loss values for both output_modes
            logits = model(input_ids, segment_ids, input_mask, labels=None)
            #print(logits, logits.shape)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            # tr_loss += loss.item()
            tr_loss.append(loss.item())
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            # print("Training loss: ", tr_loss, nb_tr_examples)
            print('Training loss: {:.4f}'.format(np.mean(tr_loss)))
            # print(f'Training loss: {np.mean(tr_loss):.4f}')
            # validation
            
            if (step + 1) % args.valid_step == 0: # TODO: number has to be checked
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

                # If we save using the predefined names, we can load using `from_pretrained`          
                path = os.path.join(args.output_dir, f'epoch_{num_train_epoch:04d}_step{step:05d}')
                mkdir(path)
                output_model_file = os.path.join(path, WEIGHTS_NAME)
                output_config_file = os.path.join(path, CONFIG_NAME)

                torch.save(model_to_save.state_dict(), output_model_file)
                model_to_save.config.to_json_file(output_config_file)
                tokenizer.save_vocabulary(path)
                
def main():
    # log_file_path = os.path.join(os.path.join(args.root_path, "log"), f'3part_{args.dataset}_train_{args.k}_log.txt')
    log_file_path = os.path.join(args.root_path, 'log', f'3part_{args.dataset}_train_{args.k}_log.txt')
    sys.stdout = Logger(log_file_path)
    train(args)

if __name__ == '__main__':
    main()