import random
from tqdm import tqdm

# def get_similar():

def create_entity_dicts(all_tuples):
    e1_to_multi_e2 = {}
    e2_to_multi_e1 = {}

    for tup in all_tuples:
        e1, rel, e2 = tup

        if (e1, rel) in e1_to_multi_e2:
            e1_to_multi_e2[(e1, rel)].append(e2)
        else:
            e1_to_multi_e2[(e1, rel)] = [e2]

        if (e2, rel) in e1_to_multi_e2:
            e1_to_multi_e2[(e2, rel)].append(e1)
        else:
            e2_to_multi_e1[(e2, rel)] = [e1]

        if (e2, rel) in e2_to_multi_e1:
            e2_to_multi_e1[(e2, rel)].append(e1)
        else:
            e2_to_multi_e1[(e2, rel)] = [e1]
 
    return e1_to_multi_e2, e2_to_multi_e1

def load_data():
    train_triples = []
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
    return valid_triples, valid_entity_list, test_triples, test_entity_list, all_triples_str_set, \
        e1_to_multi_e2, e2_to_multi_e1

def generate_head_tail(data_type, triples, entity_list, e2_to_multi_e1,e1_to_multi_e2, all_triples_str_set):
    for i, test_triple in enumerate(tqdm(triples)):
        head = test_triple[0]
        relation = test_triple[1]
        tail = test_triple[2]
        #print(test_triple, head, relation, tail)

        multi_head = e2_to_multi_e1[(tail, relation)]
        multi_tail = e1_to_multi_e2[(head, relation)]

        head_corrupt_str = [head + '\t' + relation + '\t' + tail + '\t' + '1' + '\t' + 'head' + '\t' + str(i)]
        for corrupt_ent in entity_list:
            if corrupt_ent != head and corrupt_ent not in multi_head:
                tmp_triple = [corrupt_ent, relation, tail]
                tmp_triple_str = '\t'.join(tmp_triple)
                if tmp_triple_str not in all_triples_str_set:
                    tmp_triple_str = tmp_triple_str + '\t' + '1' + '\t' + 'head' + '\t' + str(i)
                    head_corrupt_str.append(tmp_triple_str)


        tail_corrupt_str = [head + '\t' + relation + '\t' + tail + '\t' + '1' + '\t' + 'tail' + '\t' + str(i)]
        for corrupt_ent in entity_list:
            if corrupt_ent != tail and corrupt_ent not in multi_tail:
                tmp_triple = [head, relation, corrupt_ent]
                tmp_triple_str = '\t'.join(tmp_triple)
                if tmp_triple_str not in all_triples_str_set:
                    tmp_triple_str = tmp_triple_str + '\t' + '1' + '\t' + 'tail' + '\t' + str(i)
                    tail_corrupt_str.append(tmp_triple_str)

        with open("./" + data_type + "_head.txt","a") as f:
            for i in range(len(head_corrupt_str)):
                f.write(head_corrupt_str[i] + "\n")
        with open("./" + data_type + "_tail.txt","a") as f:
            for i in range(len(tail_corrupt_str)):
                f.write(tail_corrupt_str[i] + "\n")

def generate_random_head_tail(data_type, triples, entity_list, e2_to_multi_e1,e1_to_multi_e2, all_triples_str_set):
    for i, test_triple in enumerate(tqdm(triples)):
        head = test_triple[0]
        relation = test_triple[1]
        tail = test_triple[2]
        #print(test_triple, head, relation, tail)

        multi_head = e2_to_multi_e1[(tail, relation)]
        multi_tail = e1_to_multi_e2[(head, relation)]

        head_corrupt_str = [head + '\t' + relation + '\t' + tail + '\t' + '1' + '\t' + 'head' + '\t' + str(i)]
        tmp_list = random.sample(entity_list,500)
        count = 0
        for corrupt_ent in tmp_list:
            if count > 398: break
            if corrupt_ent != head and corrupt_ent not in multi_head:
                tmp_triple = [corrupt_ent, relation, tail]
                tmp_triple_str = '\t'.join(tmp_triple)
                if tmp_triple_str not in all_triples_str_set:
                    count += 1
                    tmp_triple_str = tmp_triple_str + '\t' + '1' + '\t' + 'head' + '\t' + str(i)
                    head_corrupt_str.append(tmp_triple_str)


        tail_corrupt_str = [head + '\t' + relation + '\t' + tail + '\t' + '1' + '\t' + 'tail' + '\t' + str(i)]
        tmp_list = random.sample(entity_list,500)
        count = 0
        for corrupt_ent in entity_list:
            if count > 398: break
            if corrupt_ent != tail and corrupt_ent not in multi_tail:
                tmp_triple = [head, relation, corrupt_ent]
                tmp_triple_str = '\t'.join(tmp_triple)
                if tmp_triple_str not in all_triples_str_set:
                    count += 1
                    tmp_triple_str = tmp_triple_str + '\t' + '1' + '\t' + 'tail' + '\t' + str(i)
                    tail_corrupt_str.append(tmp_triple_str)

        with open("./" + data_type + "_random_head.txt","a") as f:
            for i in range(len(head_corrupt_str)):
                f.write(head_corrupt_str[i] + "\n")
        with open("./" + data_type + "_random_tail.txt","a") as f:
            for i in range(len(tail_corrupt_str)):
                f.write(tail_corrupt_str[i] + "\n")

def main():
    valid_triples, valid_entity_list, test_triples, test_entity_list, all_triples_str_set, \
        e1_to_multi_e2, e2_to_multi_e1 = load_data()
    # generate_random_head_tail("valid", valid_triples, valid_entity_list, e2_to_multi_e1,e1_to_multi_e2, all_triples_str_set)
    # generate_random_head_tail("test", test_triples, test_entity_list, e2_to_multi_e1,e1_to_multi_e2, all_triples_str_set)
    generate_head_tail("valid", valid_triples, valid_entity_list, e2_to_multi_e1,e1_to_multi_e2, all_triples_str_set)
    generate_head_tail("test", test_triples, test_entity_list, e2_to_multi_e1,e1_to_multi_e2, all_triples_str_set)
    print('main')
if __name__ == '__main__':
    try:
        main()
        print("OK!")
    except KeyboardInterrupt:
        print('Interrupted')