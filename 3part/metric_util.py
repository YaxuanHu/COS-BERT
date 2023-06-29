import numpy as np
# from parse import args

def get_rank(preds, indices_list):
    """
    preds: (N,)
    indices_list: list of indices
    """
    ranks = []
    for indices in indices_list:
        sorted_index = np.argsort(preds[indices])[::-1]
        pos_rank = np.where(sorted_index == 0)[0]
        ranks.append(pos_rank)
    return np.concatenate(ranks)

def get_3part_rank(preds):
    """
    preds: (N,)
    indices_list: list of indices
    """
    ranks = []
    for pred in preds:
        sorted_index = np.argsort(pred)[::-1]
        pos_rank = np.where(sorted_index == 0)[0]
        ranks.append(pos_rank)
    return np.concatenate(ranks)

def hit(ranks, k):
    hit = np.mean(ranks < k)
    return hit

def count_hit(ranks, k):
    hit = np.count_nonzero(ranks < k)
    return hit

def mr(ranks):
    return np.mean(ranks+1)

def mrr(ranks):
    return np.mean(1/(ranks+1))
    
def mrr_3part(ranks, data_length):
    return (np.sum(1/(ranks+1)))/data_length/2

def stat_metric(ranks, ks=(1, 3, 10)):
    for k in ks:
        print(f'Hits@{k}: {hit(ranks, k):.4f}')
    print(f'MR: {mr(ranks):.4f}')
    print(f'MRR: {mrr(ranks):.4f}')

def stat_3part_metric(ranks, data_length, ks=(1, 3, 10)):
    for k in ks:
        print(f'Hits@{k}: {count_hit(ranks, k)/data_length/2:.4f}')
    # print(f'MR: {mr(ranks):.4f}')
    print(f'MRR: {mrr_3part(ranks, data_length):.4f}')