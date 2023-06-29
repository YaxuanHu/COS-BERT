import pickle

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

types = ["train", "valid", "test"]

def get_node_list():
    output = []
    for i, type in enumerate(types):
        datafile = load_obj("./data/"+ type + "_node2id")
        output.extend(list(datafile.keys()))
    output = list(set(output))
    # write into file
    with open("nodelist.txt","a") as f:
        for i in range(len(output)):
            f.write(output[i] + "\n") 
            
def get_relation_list():
    output = []
    for i, type in enumerate(types):
        datafile = load_obj("./data/"+ type + "_id2relation")
        output.extend(list(datafile.values()))
    output = list(set(output))
    # write into file
    with open("relationlist.txt","a") as f:
        for i in range(len(output)):
            f.write(output[i] + "\n") 

def main():
    # get_node_list()
    get_relation_list()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
