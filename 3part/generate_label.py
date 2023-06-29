import random
from tqdm import tqdm
import multiprocessing

def create_examples(count, set_type, lines, lines_str_set, entities):
    # print("count: ",count, "set_type: ", set_type, "-----------1------------")
    examples_right = []
    examples_head = []
    examples_tail = []
    for (i, line) in tqdm(enumerate(lines)):
        head_ent_text = line[0]
        tail_ent_text = line[2]
        relation_text = line[1]

        if set_type == "xxx":
            label = "1"
            # guid = "%s-%s" % (set_type, i)
            text_a = head_ent_text
            text_b = relation_text
            text_c = tail_ent_text 
            examples_right.append(text_a + "\t" + text_b + "\t" + text_c + "\t" + "1")
            
        elif set_type == "train" or set_type == "valid" or set_type == "test":
            # guid = "%s-%s" % (set_type, i)
            text_a = head_ent_text
            text_b = relation_text
            text_c = tail_ent_text 
            examples_head.append(text_a + "\t" + text_b + "\t" + text_c + "\t" + "1" + "\t" + "head")
            examples_tail.append(text_a + "\t" + text_b + "\t" + text_c + "\t" + "1" + "\t" + "tail")

            rnd = random.random()
            # guid = "%s-%s" % (set_type + "_corrupt", i)
            if rnd <= 0.5:
                # corrupting head
                for j in range(10):
                    tmp_head = ''
                    while True:
                        tmp_ent_list = set(entities)
                        tmp_ent_list.remove(line[0])
                        tmp_ent_list = list(tmp_ent_list)
                        tmp_head = random.choice(tmp_ent_list)
                        if tmp_head:
                            tmp_triple_str = tmp_head + '\t' + line[1] + '\t' + line[2]
                            if tmp_triple_str not in lines_str_set:
                                break
                    examples_head.append(tmp_head + "\t" + text_b + "\t" + text_c + "\t" + "0" + "\t" + "head")            
                    # examples.append(
                    #     InputExample(guid=guid, text_a=tmp_head, text_b=text_b, text_c = text_c, label="0"))       
            else:
                # corrupting tail
                tmp_tail = ''
                for j in range(10):
                    while True:
                        tmp_ent_list = set(entities)
                        tmp_ent_list.remove(line[2])
                        tmp_ent_list = list(tmp_ent_list)
                        tmp_tail = random.choice(tmp_ent_list)
                        if tmp_tail:
                            tmp_triple_str = line[0] + '\t' + line[1] + '\t' + tmp_tail
                            if tmp_triple_str not in lines_str_set:
                                break
                    examples_tail.append(text_a + "\t" + text_b + "\t" + tmp_tail + "\t" + "0" + "\t" + "tail")  
                    # examples.append(
                    #     InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c = tmp_tail, label="0")) 
    # print("count: ",count, "set_type: ", set_type, "-----------2------------")
    # with open("./data_label/" + set_type + "_right_label" +str(count) +".txt","a") as f:
        # i = 0 
        # for i in range(len(examples_right)):
        #     f.write(examples_right[i] + "\n")
    with open("./data_label/" + set_type + "_head_label" +str(count) +".txt","a") as f:
        for i in range(len(examples_head)):
            f.write(examples_head[i] + "\n")
    with open("./data_label/" + set_type + "_tail_label" +str(count) +".txt","a") as f:
        for i in range(len(examples_tail)):
            f.write(examples_tail[i] + "\n")
    # print("count: ",count, "set_type: ", set_type, "-----------3------------")                                              

def main():


    lines = []
    lines_str_set = set()
    # read triple list

    # read nodelist
    nodelistfile = open("nodelist.txt","r")
    nodelist = nodelistfile.read()
    entities = nodelist.split("\n")

    # set_types = ["valid", "test", "train"]
    set_types = ["train"]
    # set_types = ["valid", "test"]
    # set_type = "train"
    for i, set_type in enumerate(set_types):
        with open(set_type + "_triple.txt") as f:
            data = f.readlines()
        for inst in data:
            line = []
            inst = inst.strip()
            lines_str_set.add(inst)
            inst = inst.split('\t')
            src, rel, tgt = inst
            line.append(src)
            line.append(rel)
            line.append(tgt)
            lines.append(line)

        n = 25
        pool = multiprocessing.Pool(processes=n)
        lines_len = len(lines)
        t = lines_len//n
        i = 0
        begin = 0
        end = 0

        for i in tqdm(range(n+1)):
            begin = i * t
            end = (i + 1) * t
            if end > lines_len:
                end = lines_len
            tmp_lines = lines[begin:end]
            pool.apply_async(create_examples, (i, set_type, tmp_lines, lines_str_set, entities))
        pool.close()
        pool.join()

if __name__ == '__main__':
    main()