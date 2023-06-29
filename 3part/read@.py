import numpy as np
import logging
logger = logging.getLogger(__name__)
import pickle
def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def append_all(i, acc0, acc1, acc2, acc3):
    list = []
    if i < 3:
        list.extend(acc0)
        list.extend(acc1)
        list.extend(acc2)
        list.extend(acc3)
    else:
        hits_level = 0
        for hits_level in range(10):
            list.append([])
        hits_level = 0
        for hits_level in range(10):
            list[hits_level].extend(acc0[hits_level])
            list[hits_level].extend(acc1[hits_level])
            list[hits_level].extend(acc2[hits_level])
            list[hits_level].extend(acc3[hits_level])
    return list

def load_data():
    path = './output/64epoch5/'
    acc_types = ['ranks', 'ranks_left', 'ranks_right', 'hits', 'hits_left', 'hits_right']

    ranks = []
    ranks_left = []
    ranks_right = []

    hits = []
    hits_left = []
    hits_right = []

    for j in range(10):
        hits_left.append([])
        hits_right.append([])
        hits.append([])
    

    for i in range(len(acc_types)):
        acc_type = acc_types[i]
        acc0 = load_obj(path+acc_type+'/0')
        acc1 = load_obj(path+acc_type+'/1')
        acc2 = load_obj(path+acc_type+'/2')
        acc3 = load_obj(path+acc_type+'/3')
        if i == 0: ranks = append_all(i, acc0, acc1, acc2, acc3)
        elif i == 1: ranks_left = append_all(i, acc0, acc1, acc2, acc3)
        elif i == 2: ranks_right = append_all(i, acc0, acc1, acc2, acc3)
        elif i == 3: hits = append_all(i, acc0, acc1, acc2, acc3)
        elif i == 4: hits_left = append_all(i, acc0, acc1, acc2, acc3)
        elif i == 5: hits_right = append_all(i, acc0, acc1, acc2, acc3)
    return ranks, ranks_left, ranks_right, hits, hits_left, hits_right


def main():
    ranks = []
    ranks_left = []
    ranks_right = []

    hits = []
    hits_left = []
    hits_right = []

    for j in range(10):
        hits_left.append([])
        hits_right.append([])
        hits.append([])

    ranks, ranks_left, ranks_right, hits, hits_left, hits_right = load_data()
    # print(len(hits))

    # for i in [0,2,9]:
    #     logger.info('Hits left @{0}: {1}'.format(i+1, np.mean(hits_left[i])))
    #     logger.info('Hits right @{0}: {1}'.format(i+1, np.mean(hits_right[i])))
    #     logger.info('Hits @{0}: {1}'.format(i+1, np.mean(hits[i])))
    # logger.info('Mean rank left: {0}'.format(np.mean(ranks_left)))
    # logger.info('Mean rank right: {0}'.format(np.mean(ranks_right)))
    # logger.info('Mean rank: {0}'.format(np.mean(ranks)))
    # logger.info('Mean reciprocal rank left: {0}'.format(np.mean(1./np.array(ranks_left))))
    # logger.info('Mean reciprocal rank right: {0}'.format(np.mean(1./np.array(ranks_right))))
    # logger.info('Mean reciprocal rank: {0}'.format(np.mean(1./np.array(ranks))))   
    for i in [0,2,9]:
        print('Hits left @{0}: {1}'.format(i+1, np.mean(hits_left[i])))
        print('Hits right @{0}: {1}'.format(i+1, np.mean(hits_right[i])))
        print('Hits @{0}: {1}'.format(i+1, np.mean(hits[i])))
    print('Mean rank left: {0}'.format(np.mean(ranks_left)))
    print('Mean rank right: {0}'.format(np.mean(ranks_right)))
    print('Mean rank: {0}'.format(np.mean(ranks)))
    print('Mean reciprocal rank left: {0}'.format(np.mean(1./np.array(ranks_left))))
    print('Mean reciprocal rank right: {0}'.format(np.mean(1./np.array(ranks_right))))
    print('Mean reciprocal rank: {0}'.format(np.mean(1./np.array(ranks))))   



if __name__ == '__main__':
    try:
        main()
        print("OK!")
    except KeyboardInterrupt:
        print('Interrupted')