import codecs
import pickle
def read_data( input_file):
    """Reads a BIO data."""
    with codecs.open(input_file, 'r', encoding='utf-8') as f:
        lines = []
        words = []
        labels = []
        for line in f:
            contends = line.strip()
            tokens = contends.split(' ')
            if len(tokens) == 2:
                word = line.strip().split(' ')[0]
                label = line.strip().split(' ')[-1]
            else:
                if len(contends) == 0:
                    l = ' '.join([label for label in labels if len(label) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])
                    lines.append([l, w])
                    words = []
                    labels = []
                    continue
            if contends.startswith("-DOCSTART-"):
                words.append('')
                continue
            words.append(word)
            labels.append(label)
        return lines


def loadData(openPath):
    with open(openPath, "rb", ) as file:
        data=pickle.load(file)
        return data

def dumpData4Gb(data,openPath):
    with open(openPath, "wb" ) as file:
        pickle.dump(data,file,protocol=4)


#对于ndarray 来处理
def minibatchesNdArray(test_input_x,test_input_label,test_input_len,test_token, minibatch_size):
    iterNum=len(test_input_x)//minibatch_size
    for i in range(iterNum):
        start=i*minibatch_size
        end=(i+1)*minibatch_size
        yield test_input_x[start:end],test_input_label[start:end],test_input_len[start:end],test_token[start:end]

