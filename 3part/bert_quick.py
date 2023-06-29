import data.data_utils as data_utils
import models.memn2n as memn2n

from sklearn import metrics
import numpy as np

import argparse

import tensorflow as tf

import pickle as pkl
import sys
import io
import os
#import jieba
#import jieba.analyse as jan

#sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
sys.path.append('BERT/')

import modeling
import tokenization
from run_classifier import model_fn_builder, InputExample, convert_examples_to_features

DATA_DIR = 'data/'
P_DATA_DIR = 'data/processed/'
BATCH_SIZE = 32
CKPT_DIR= 'ckpt/'

'''
    dictionary of models
        select model from here
model = {
        'memn2n' : memn2n.MemN2NDialog
        }# add models, as you implement

'''

def get_dict_list(data_dir):
    file_path = os.path.join(data_dir, 'task1-dict.txt')
    with open(file_path, 'r', encoding='utf-8') as f:
      reader = f.readlines()
      d = {}
      for index, line in enumerate(reader):
        split_line = line.strip().split('\t')
        d[split_line[0]] = split_line[1]
    return d

def get_labels(data_dir):
    file_path = os.path.join(data_dir, 'task1-dict.txt')
    num_labels = 0
    with open(file_path, 'r', encoding='utf-8') as f:
      reader = f.readlines()
      for _ in enumerate(reader):
        num_labels += 1
    return [str(i) for i in range(num_labels)]

def get_input_example(guid, text_a):
    examples = []
    examples.append(InputExample(guid=guid, text_a=text_a,
                               text_b=None, label='0'))
    return examples

class FastPredict:

    def __init__(self, estimator):
        self.estimator = estimator
        self.first_run = True
        self.closed = False
        self.query = None

    def input_fn_builder(self, seq_length=128, is_training=False,
                                drop_remainder=False):
        """Creates an `input_fn` closure to be passed to TPUEstimator."""

        vocab_file = 'chinese/vocab.txt'
        label_list = get_labels('BERT/input')
        tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=True)

        def input_fn():
            """The actual input function."""
            all_input_ids = []
            all_input_mask = []
            all_segment_ids = []
            all_label_ids = []
            all_is_real_example = []

            for feature in features:
                all_input_ids.append(feature.input_ids)
                all_input_mask.append(feature.input_mask)
                all_segment_ids.append(feature.segment_ids)
                all_label_ids.append(feature.label_id)
                all_is_real_example.append(feature.is_real_example)

            return {'input_ids': all_input_ids,
                    'input_mask': all_input_mask,
                    'segment_ids': all_segment_ids,
                    'label_ids': all_label_ids,
                    'is_real_example': all_is_real_example}

        while not self.closed:
            if not self.query:
                self.query = '<silence>'
            predict_examples = get_input_example('test', self.query)
            features = convert_examples_to_features(
                                predict_examples, label_list,
                                128, tokenizer)
            yield input_fn()

    def input_fn(self):
        return (tf.data.Dataset.from_generator(
            self.input_fn_builder,
            output_types={'input_ids':tf.int32,
                          'input_mask':tf.int32,
                          'segment_ids':tf.int32,
                          'label_ids':tf.int32,
                          'is_real_example':tf.int32},
            output_shapes={'input_ids':(None,128),
                           'input_mask':(None,128),
                           'segment_ids':(None,128),
                           'label_ids':(None),
                           'is_real_example':(None)})) 
 
    def predict(self, feature_batch):
        """ Runs a prediction on a set of features. Calling multiple times
            does *not* regenerate the graph which makes predict much faster.
            feature_batch a list of list of features. IMPORTANT: If you're only classifying 1 thing,
            you still need to make it a batch of 1 by wrapping it in a list (i.e. predict([my_feature]), not predict(my_feature) 
        """
        self.next_features = feature_batch
        if self.first_run:
            self.batch_size = len(feature_batch)
            self.predictions = self.estimator.predict(
                input_fn=self.input_fn)
            next(self.predictions)
            self.first_run = False
        elif self.batch_size != len(feature_batch):
            raise ValueError("All batches must be of the same size. First-batch:" + str(self.batch_size) + " This-batch:" + str(len(feature_batch)))

        results = []
        for _ in range(self.batch_size):
            self.query = feature_batch[0]
            results.append(next(self.predictions))
        return results

    def close(self):
        self.closed=True
        next(self.predictions)

'''
    run prediction on dataset

'''
def batch_predict(model, S,Q,n, batch_size):
    preds = []
    for start in range(0, n, batch_size):
        end = start + batch_size
        s = S[start:end]
        q = Q[start:end]
        pred = model.predict(s, q)
        preds += list(pred)
    return preds


'''
    preprocess data

'''
def prepare_data(args):
    # get candidates (restaurants)
    candidates, candid2idx, idx2candid = data_utils.load_candidates(
            candidates_f= DATA_DIR + 'candidates.txt')
    # get data
    train, test, val = data_utils.load_dialog_task(
            data_dir= DATA_DIR,
            candid_dic= candid2idx,
            isOOV= False)
    ##
    # get metadata
    metadata = data_utils.build_vocab(train + test + val, candidates)

    ###
    # write data to file
    data_ = {
            'candidates' : candidates,
            'train' : train,
            'test' : test,
            'val' : val
            }
    with open(P_DATA_DIR + 'data.pkl', 'wb') as f:
        pkl.dump(data_, f)

    ### 
    # save metadata to disk
    metadata['candid2idx'] = candid2idx
    metadata['idx2candid'] = idx2candid

    with open(P_DATA_DIR + 'metadata.pkl', 'wb') as f:
        pkl.dump(metadata, f)


'''
    parse arguments

'''
def parse_args(args):
    parser = argparse.ArgumentParser(
            description='Train Model for Goal Oriented Dialog Task')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-i', '--infer', action='store_true',
                        help='perform inference in an interactive session')
    group.add_argument('--ui', action='store_true',
                        help='interact through web app(flask); do not call this from cmd line')
    group.add_argument('-t', '--train', action='store_true',
                        help='train model')
    group.add_argument('-d', '--prep_data', action='store_true',
                        help='prepare data')
    parser.add_argument('--batch_size', required=False, type=int, default=16,
                        help='you know what batch size means!')
    parser.add_argument('--epochs', required=False, type=int, default=100,
                        help='num iteration of training over train set')
    parser.add_argument('--eval_interval', required=False, type=int, default=5,
                        help='num iteration of training over train set')
    parser.add_argument('--log_file', required=False, type=str, default='log.txt',
                        help='enter the name of the log file')


    parser.add_argument(
    "--data_dir", required=False, type=str, default='BERT/input/',
    help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument(
    "--bert_config_file", required=False, type=str, default='chinese/bert_config.json', 
    help="The config json file corresponding to the pre-trained BERT model. This specifies the model architecture.")
    parser.add_argument(
    "--vocab_file", required=False, type=str, default='chinese/vocab.txt',
    help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument(
    "--output_dir", required=False, type=str, default='BERT/output/',
    help="The output directory where the model checkpoints will be written.")

    parser.add_argument(
    "--init_checkpoint", required=False, type=str, default='chinese/bert_model.ckpt',
    help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument(
    "--max_seq_length", required=False, type=int, default=128,
    help="The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

    args = vars(parser.parse_args(args))
    return args


class InteractiveSession():

    def __init__(self, model, idx2candid, w2idx, n_cand, memory_size):
        self.context = []
        self.u = None
        self.r = None
        self.nid = 1
        self.model = model
        self.idx2candid = idx2candid
        self.w2idx = w2idx
        self.n_cand = model._candidates_size
        self.memory_size = memory_size
        self.model = model

    def reply(self, msg):
        line = msg.strip().lower()
        if line == 'clear':
            self.context = []
            self.nid = 1
            reply_msg = 'memory cleared!'
        else:
            u = data_utils.tokenize(line, False)
            data = [(self.context, u, -1)]
            s, q, a = data_utils.vectorize_data(data,
                    self.w2idx,
                    self.model._sentence_size,
                    1,
                    self.n_cand,
                    self.memory_size)
            preds = self.model.predict(s,q)
            r = self.idx2candid[preds[0]]
            #reply_msg = jan.textrank(r, allowPOS=('n','v','m','l','an','r'))
            reply_msg = r
            r = data_utils.tokenize(r, True)
            u.append('$u')
            u.append('#'+str(self.nid))
            r.append('$r')
            r.append('#'+str(self.nid))
            self.context.append(u)
            self.context.append(r)
            self.nid+=1
        return reply_msg



def main(args):
    # parse args
    args = parse_args(args)

    # prepare data
    if args['prep_data']:
        print('\n>> Preparing Data\n')
        prepare_data(args)
        sys.exit()

    # ELSE
    # read data and metadata from pickled files
    with open(P_DATA_DIR + 'metadata.pkl', 'rb') as f:
        metadata = pkl.load(f)
    with open(P_DATA_DIR + 'data.pkl', 'rb') as f:
        data_ = pkl.load(f)

    # read content of data and metadata
    candidates = data_['candidates']
    candid2idx, idx2candid = metadata['candid2idx'], metadata['idx2candid']

    # get train/test/val data
    train, test, val = data_['train'], data_['test'], data_['val']

    # gather more information from metadata
    sentence_size = metadata['sentence_size']
    w2idx = metadata['w2idx']
    idx2w = metadata['idx2w']
    memory_size = metadata['memory_size']
    vocab_size = metadata['vocab_size']
    n_cand = metadata['n_cand']
    candidate_sentence_size = metadata['candidate_sentence_size']

    # vectorize candidates
    candidates_vec = data_utils.vectorize_candidates(candidates, w2idx, candidate_sentence_size)

    ###
    # create model
    #model = model['memn2n']( # why?
    model = memn2n.MemN2NDialog(
                batch_size= BATCH_SIZE,
                vocab_size= vocab_size,
                candidates_size= n_cand,
                sentence_size= sentence_size,
                embedding_size= 20,
                candidates_vec= candidates_vec,
                hops= 3
            )
    # gather data in batches
    train, val, test, batches = data_utils.get_batches(train, val, test, metadata, batch_size=BATCH_SIZE)

    if args['train']:
        # training starts here
        epochs = args['epochs']
        eval_interval = args['eval_interval']
        #
        # training and evaluation loop
        print('\n>> Training started!\n')
        # write log to file
        log_handle = open('log/' + args['log_file'], 'w')
        cost_total = 0.
        #best_validation_accuracy = 0.
        for i in range(epochs+1):

            #print('epochs'+str(i))
            for start, end in batches:
                s = train['s'][start:end]
                q = train['q'][start:end]
                a = train['a'][start:end]
                cost_total += model.batch_fit(s, q, a)

            if i%eval_interval == 0 and i:
                train_preds = batch_predict(model, train['s'], train['q'], len(train['s']), batch_size=BATCH_SIZE)
                val_preds = batch_predict(model, val['s'], val['q'], len(val['s']), batch_size=BATCH_SIZE)
                train_acc = metrics.accuracy_score(np.array(train_preds), train['a'])
                val_acc = metrics.accuracy_score(val_preds, val['a'])
                print('Epoch[{}] : <ACCURACY>\n\ttraining : {} \n\tvalidation : {}'.
                     format(i, train_acc, val_acc))
                log_handle.write('{} {} {} {}\n'.format(i, train_acc, val_acc,
                    cost_total/(eval_interval*len(batches))))
                cost_total = 0. # empty cost
                #
                # save the best model, to disk
                #if val_acc > best_validation_accuracy:
                #best_validation_accuracy = val_acc
                model.saver.save(model._sess, CKPT_DIR + 'memn2n_model.ckpt', global_step=i)
        # close file
        log_handle.close()

    else: # inference
        ###
        # restore checkpoint
        ckpt = tf.train.get_checkpoint_state(CKPT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            print('\n>> restoring checkpoint from', ckpt.model_checkpoint_path)
            model.saver.restore(model._sess, ckpt.model_checkpoint_path)
        #  interactive(model, idx2candid, w2idx, sentence_size, BATCH_SIZE, n_cand, memory_size)

        # create an interactive session instance
        isess = InteractiveSession(model, idx2candid, w2idx, n_cand, memory_size)

        bert_config = modeling.BertConfig.from_json_file(args['bert_config_file'])

        label_list = get_labels(args['data_dir'])

        tokenizer = tokenization.FullTokenizer(
            vocab_file=args['vocab_file'], do_lower_case=True)

        tpu_cluster_resolver = None

        #is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        #run_config = tf.contrib.tpu.RunConfig(
        #    cluster=tpu_cluster_resolver,
        #    master=None,
        #    model_dir=output_dir,
        #    save_checkpoints_steps=1000,
        #    tpu_config=tf.contrib.tpu.TPUConfig(
        #        iterations_per_loop=1000,
        #        num_shards=8,
        #        per_host_input_for_training=is_per_host))
        run_config = tf.estimator.RunConfig(
            model_dir=args['output_dir'],
            save_checkpoints_steps=1000)
        model_fn = model_fn_builder(
            bert_config=bert_config,
            num_labels=len(label_list),
            init_checkpoint=args['init_checkpoint'],
            learning_rate=5e-5,
            num_train_steps=100,
            num_warmup_steps=100,
            use_tpu=False,
            use_one_hot_embeddings=False)

        # If TPU is not available, this will fall back to normal Estimator on CPU
        # or GPU.
        # estimator = tf.contrib.tpu.TPUEstimator(
        #     use_tpu=False,
        #     model_fn=model_fn,
        #     config=run_config,
        #     train_batch_size=16,
        #     eval_batch_size=8,
        #     predict_batch_size=8)
        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config)

        fast_estimator = FastPredict(estimator)

        d = get_dict_list(args['data_dir'])

        fast_estimator.predict(['startup'])

        if args['infer']:
            query = ''
            while query != 'exit':
                if query and query != 'clear':
                    result = fast_estimator.predict([query])
                    #for prediction in result:
                    prediction = result[0]['probabilities']
                    query = d[str(np.argmax(prediction))]

                reply = str(isess.reply(query))
                print('- ' + query)
                print(':: ' + reply)
                if query == 'clear':
                    print('>> ')
                    query = ''
                else:
                    query = input('>> ')
        elif args['ui']:
            return isess

# _______MAIN_______
if __name__ == '__main__':
    #jieba.initialize()
    #jan.set_idf_path('data/tfidf.big')
    main(sys.argv[1:])
    #main(['--infer', '--task_id=1'])

