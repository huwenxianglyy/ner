import utils
from blstm import  BLSTM
import os
import tensorflow as tf
import numpy as np
from NEREvaluateUtile import evaluateUtils
from tensorflow.contrib.layers.python.layers import initializers


flags = tf.flags

FLAGS = flags.FLAGS




os.environ['CUDA_VISIBLE_DEVICES'] = '0'





flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization."
)


flags.DEFINE_float("learning_rate", 0.001, "The initial learning rate for Adam.")
flags.DEFINE_float("num_train_epochs", 100.0, "Total number of training epochs to perform.")
flags.DEFINE_float('droupout_rate', 0.5, 'Dropout rate')
flags.DEFINE_float('clip', 5, 'Gradient clip')
flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")


flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")



tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

# lstm parame
flags.DEFINE_integer('lstm_size', 256, 'size of lstm units')
flags.DEFINE_integer('num_layers', 1, 'number of rnn layers, default is 1')
flags.DEFINE_integer("batch_size",60,"number of batch")

flags.DEFINE_string('cell', 'lstm', 'which rnn cell used')


flags.DEFINE_list("label_list",['O', 'B-ORG', 'B-LOC', 'I-PER', 'I-ORG', 'I-LOC', 'B-PER'],"label list") #


label_map = {}
# 对label进行index化
for (i, label) in enumerate(FLAGS.label_list):
    label_map[label] = i
flags.DEFINE_list("label_map",[label_map],"label_map list") # 因为flag不能定义字典，就用list代替

flags.DEFINE_integer('num_labels', len(label_map), 'number of rnn labels')


def evaulateTest( model,sess, test_input_x,test_input_label,test_input_len,test_token, e,
                 idx_to_tag, combine_type_list=None):
    '''
    :param model:  模型
    :param sess:  session 会话
    :param test_input_x:  input_x
    :param test_input_label: input_label
    :param test_input_len: input_len
    :param test_token: 原文
    :param e: 评估对象
    :param idx_to_tag:
    :param combine_type_list: 一般输入[place,org] 表示需要将place 和 org 合并，并以最后一个标签为最终标签，例如 上海红心厂 会被标注成 place和org ，但其实这里是个org 所以这里需要将place org合并
    :return:
    '''


    loss = []
    predictlist = []
    real_labels = []
    tokens = []
    total_lengths = []
    for test_x_batch, test_labels_batch,test_len_batch, test_token_batch in utils.minibatchesNdArray(
            test_input_x, test_input_label, test_input_len, test_token, FLAGS.batch_size):
        feed = getFeed(test_x_batch, test_len_batch, test_labels_batch)
        los, pred, lengths = predict(model,sess, feed)
        loss.append(los)
        tokens.extend(test_token_batch.tolist())
        predictlist.extend(pred.tolist())
        real_labels.extend(test_labels_batch.tolist())
        total_lengths.extend(lengths.tolist())
    # 这里打印测试的信息
    print("测试集合loss为:%.3f" % np.mean(loss))
    # 这里是train的评估
    pred = split(predictlist, total_lengths)
    real_label = split(real_labels, total_lengths)
    e.evaulateOflogits(pred, real_label, idx_to_tag, combine_type_list, tokens, False)

def split(label,length):
    result=[]
    for i,l in enumerate(label):
        result.append(l[1:length[i]-1])
    return  result




def predict(model,sess,feed):
    los, pred, lengths = sess.run([model.totle_loss, model.predict, model.lengths], feed_dict=feed)
    return los,pred,lengths

def createInPut(inputDatas,max_seq_length,word2Index,label_map):


    inputX_Array=[]
    label_Array=[]
    tokens_Array=[]
    seq_len=[]
    for input in  inputDatas:
        texts=input[1].split(" ")
        labels = input[0].split(" ")
        w_id=[word2Index[w] if w in word2Index  else word2Index["[UNK]"] for w in texts][0:max_seq_length]
        label_id=[label_map[l] for l in labels][0:max_seq_length]
        texts=texts[0:max_seq_length]
        seq_len.append(len(texts))
        while len(w_id) < max_seq_length:
            w_id.append(word2Index["[PAD]"])
            label_id.append(label_map["O"])
            texts.append("N")
        inputX_Array.append(w_id)
        label_Array.append(label_id)
        tokens_Array.append(texts)
    return np.asarray(inputX_Array,dtype=np.int32),np.asarray(label_Array,dtype=np.int32),np.asarray(seq_len,dtype=np.int32),np.asarray(tokens_Array,dtype=np.str)






is_training=True


word2vec_path="./word2vec/small.300.bin"
word = utils.loadData(word2vec_path)
word2Index=word["word2Index"]
word_vectors=word["word2Vec"]


test_data=utils.read_data("./NERdata/test.txt")
train_data=utils.read_data("./NERdata/train.txt")
train_input_x,train_input_label,train_input_len,train_token=createInPut(train_data,FLAGS.max_seq_length,word2Index,label_map)
test_input_x,test_input_label,test_input_len,test_token=createInPut(test_data,FLAGS.max_seq_length,word2Index,label_map)





num_train_steps = int(  # 这个是总的迭代次数，训练完一轮的次数*总的次数
    len(train_input_x) / FLAGS.batch_size * FLAGS.num_train_epochs)








input_x = tf.placeholder(shape=[None, FLAGS.max_seq_length], dtype=tf.int64, name="input_x")
input_labels = tf.placeholder(shape=[None,FLAGS.max_seq_length], dtype=tf.int64, name="input_y")
input_len = tf.placeholder(shape=[None], dtype=tf.int64, name="input_len")
def getFeed(x, len, labels=None):
    if labels is None:
        feed = {input_x: x,
                input_len: len
                }

    else:
        feed = {input_x: x,
                input_len: len,
                input_labels: labels}
    return feed



wordemb = tf.Variable(np.asarray(word_vectors), name="word2vec", trainable=False)
input_emb = tf.nn.embedding_lookup(wordemb, input_x)
blstm_model = BLSTM(embedded_chars=input_emb, hidden_unit=FLAGS.lstm_size, cell_type=FLAGS.cell,
                      num_layers=FLAGS.num_layers,
                      droupout_rate=FLAGS.droupout_rate, initializers=initializers, num_labels=FLAGS.num_labels,
                      seq_length=FLAGS.max_seq_length, labels=input_labels, lengths=input_len, is_training=is_training)

# blstm_model.add_blstm_layer()#    这里使用blstm
blstm_model.add_blstm_crf_layer() # 这里使用blstm+crf

train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(blstm_model.totle_loss)

init_op = tf.global_variables_initializer()
idx_to_tag = {idx: tag for tag, idx in FLAGS.label_map[0].items()}

e = evaluateUtils()
loss = []
predict_list = []



with tf.Session() as sess:
    sess.run(init_op)
    for i in range(num_train_steps):
        shuffIndex = np.random.permutation(np.arange(len(train_input_x)))
        shuffIndex=shuffIndex[0:FLAGS.batch_size]
        feed = getFeed(train_input_x[shuffIndex], train_input_len[shuffIndex], train_input_label[shuffIndex])
        _,los,pred,lengths=sess.run([train_op,blstm_model.totle_loss,blstm_model.predict,blstm_model.lengths],feed_dict=feed)
        print(i)
        if i%100==0 and i>1  :#
            # 这里测试一batch的train和全部的test
            print("训练集合loss为:%.3f"%los)

            # 这里是test的评估
            evaulateTest(blstm_model,sess, test_input_x,test_input_label,test_input_len,test_token, e,
                              idx_to_tag)
