import sys
import os
import pickle


class Logger(object):
    """
    这个类的目的是尽可能不改变原始代码的情况下, 使得程序的输出同时打印在控制台和保存在文件中
    用法: 只需在程序中加入一行 `sys.stdout = Logger(log_file_path)` 即可
    """
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

def nowdt():
    """
    get string representation of date and time of now()
    """
    from datetime import datetime

    now = datetime.now()
    return now.strftime("%d/%m/%Y %H:%M:%S")


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
		print ("---  new folder...  ---")
		print ("---  OK  ---")
	else:
		print ("---  There is this folder!  ---")

def load_file(file_name, len_file=None):
    file_list = []
    with open(file_name) as f:
        # data = f.readlines()
        # for inst in data:
        for inst in f:
            line = []
            inst = inst.strip()
            inst = inst.split('\t')
            for i, value in enumerate(inst):
                line.append(value)
            file_list.append(line)
    if len_file:
        file_list = file_list[:len_file]
    return file_list