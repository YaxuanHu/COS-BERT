from collections import Counter
from re import A
import torch
import torch.nn as nn
import numpy as np
# import argparse
import logging as logger
# from graph import Graph
import os
import pickle
from tqdm import tqdm

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def write_dic_to_file(obj, name):
    for key, value in obj.items():
        with open(name+".txt","a") as f:
            f.write(key + "\t" + value + "\n") 

def reverse(dic):
    d2 = dict()
    for k in dic.keys():
        d2[dic[k]] = k
    return d2

def reverse_relation(dic):
    d2 = dict()
    for k in dic.keys():
        d2[dic[k]] = upper_to_lower(k)
    return d2

def upper_to_lower(input_str):
    output_str = ""
    for (index, letter) in enumerate(input_str):
        if letter.isupper():
            if index > 0:
                output_str = output_str + " "
                output_str = output_str  + letter.lower()
            else:
                output_str = output_str  + letter.lower()
        else:
            output_str = output_str  + letter
    return output_str

def get_triple(type):
    datapath = type +"_triple.txt"
    save_graph = load_obj("./data/" + type + "_graph")
    save_id2node = load_obj("./data/" + type + "_id2node")
    save_id2relation = load_obj("./data/" + type + "_id2relation")
    with open(datapath,"a") as f:
        for h_id in tqdm(save_graph.keys()):
            if len(save_graph[h_id]) > 0:
                for t_id in save_graph[h_id].keys():
                    for rel_id in range(len(save_graph[h_id][t_id])):
                        f.write(save_id2node[h_id] + "\t" + save_id2relation[save_graph[h_id][t_id][rel_id]] + "\t" + save_id2node[t_id] + "\n") 


def get_quintuple(save_graph, save_id2node, save_id2relation):
    quintuple = []
    for h_id in tqdm(save_graph.keys()):
        if len(save_graph[h_id]) > 0:
            for t1_id in save_graph[h_id].keys():
                if len(save_graph[t1_id]) > 0:
                    for t2_id in save_graph[t1_id].keys():
                        for rel1_id in range(len(save_graph[h_id][t1_id])):
                            for rel2_id in range(len(save_graph[t1_id][t2_id])):
                                quintuple.append(save_id2node[h_id] + " " + save_id2relation[save_graph[h_id][t1_id][rel1_id]] 
                                                + " " + save_id2node[t1_id] + " " + save_id2relation[save_graph[t1_id][t2_id][rel2_id]]
                                                + " " + save_id2node[t2_id])
    return quintuple

class Graph:
    def __init__(self, directed=True):
        self.node2id = {}
        self.relation2id = {}
        self.edges = {}

    def find_node(self, name):
        if name in self.node2id:
            return self.node2id[name]
        else:
            return -1

    def add_node(self, name):
        self.node2id[name] = len(self.node2id)
        self.edges[self.node2id[name]] = {}
        return self.node2id[name]
    
    def find_relation(self, name):
        if name in self.relation2id:
            return self.relation2id[name]
        else:
            return -1

    def add_relation(self, name):
        self.relation2id[name] = len(self.relation2id)
        return self.relation2id[name]

    def add_edge(self, node1, node2, rel):
        # print("node1: ",node1)
        # print("node2: ",node2)
        # print("rel: ",rel)
        if node2 in self.edges[node1]:
            self.edges[node1][node2].append(rel)
        else: 
            self.edges[node1][node2] = [rel]
        # print("edges: ", self.edges[node1][node2])


class ConceptNetTSVReader:
    def __init__(self, dataset):
        logger.info("Reading ConceptNet")
        self.dataset = dataset
        self.graph = Graph()
        self.rel2id = {}

    def read_network(self, data_dir, split, train_network=None):

        if split == "train":
            data_path = os.path.join(data_dir, "train.txt")
        elif split == "valid":
            data_path = os.path.join(data_dir, "valid.txt")
        elif split == "test":
            data_path = os.path.join(data_dir, "test.txt")

        with open(data_path) as f:
            data = f.readlines()

        if split == "test":
            data = data[:1200]

        for inst in data:
            inst = inst.strip()
            if inst:
                inst = inst.split('\t')
                rel, src, tgt = inst
                src = src.lower()
                tgt = tgt.lower()
                self.add_example(src, tgt, rel)

        self.rel2id = self.graph.relation2id

    def add_example(self, src, tgt, relation):
        src_id = self.graph.find_node(src)
        if src_id == -1:
            src_id = self.graph.add_node(src)

        tgt_id = self.graph.find_node(tgt)
        if tgt_id == -1:
            tgt_id = self.graph.add_node(tgt)

        relation_id = self.graph.find_relation(relation)
        if relation_id == -1:
            relation_id = self.graph.add_relation(relation)

        edge = self.graph.add_edge(src_id, tgt_id, relation_id)

        return edge

def load_data(dataset, reader_cls, data_dir, type):
    # ------------------------------------------------
    # 重新创建图时解开注释
    train_network = reader_cls(dataset)
    train_network.read_network(data_dir=data_dir, split=type)
    save_graph = train_network.graph.edges
    # print(save_graph)
    save_node2id = train_network.graph.node2id
    save_relation2id = train_network.graph.relation2id
    save_obj(save_graph, "./data/" + type + "_graph")
    save_obj(save_node2id, "./data/"+ type + "_node2id")
    save_obj(save_relation2id, "./data/"+ type + "_relation2id")
    print(save_relation2id)
    # ------------------------------------------------
    # path = "../data/ConceptNet/"
    # save_graph = load_obj("save_graph")
    # save_node2id = load_obj("save_node2id")
    # save_relation2id = load_obj("save_relation2id")
    # save_id2relation = load_obj("save_id2relation")
    # save_id2node = load_obj("save_id2node")
    # quintuple = load_obj("quintuple")
    # quintuple = get_quintuple(save_graph, save_id2node, save_id2relation)#获得所有的五元组
    # save_obj(quintuple, "quintuple")
    save_id2node = reverse(save_node2id)
    save_obj(save_id2node, "./data/"+ type + "_id2node")
    save_id2relation = reverse_relation(save_relation2id)
    save_obj(save_id2relation, "./data/"+ type + "_id2relation")
    print(save_id2relation)
    # print("-----")
    # # print(quintuple)
    # print("-----")
    


def main():
    args_dataset = "conceptnet"
    dataset_cls = ConceptNetTSVReader
    data_dir = "../data/conceptnet-100k/"
    load_data(args_dataset,dataset_cls,data_dir, type="train")
    load_data(args_dataset,dataset_cls,data_dir, type="valid")
    load_data(args_dataset,dataset_cls,data_dir, type="test")
    get_triple("train")
    get_triple("valid")
    get_triple("test")
    print("ok")
                                                                                        
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
