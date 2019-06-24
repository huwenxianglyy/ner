import os
import utils
import tensorflow as tf
from bert import modeling
import tokenization
import numpy as np
from blstm import BLSTM
from config import FLAGS
import optimization
from tensorflow.contrib.layers.python.layers import initializers
import itertools

from NEREvaluateUtile import evaluateUtils


class ner_model(object):
    def __init__(self,is_Train):

        bert_config_path=FLAGS.bert_config_file
        vocab_file=FLAGS.vocab_file
        self.bert_config = modeling.BertConfig.from_json_file(bert_config_path)
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=FLAGS.do_lower_case)

        self.input_ids = tf.placeholder(shape=[None, FLAGS.max_seq_length], dtype=tf.int32, name="input_ids")
        self.input_mask = tf.placeholder(shape=[None, FLAGS.max_seq_length], dtype=tf.int32, name="input_mask")
        self.segment_ids = tf.placeholder(shape=[None, FLAGS.max_seq_length], dtype=tf.int32, name="segment_ids")
        self.input_labels = tf.placeholder(shape=[None, FLAGS.max_seq_length], dtype=tf.int32, name="input_labels")

        self.createModel(self.input_ids,self.input_mask,self.segment_ids,self.input_labels,is_Train)


    def split(self,label,length):
        result=[]
        for i,l in enumerate(label):
            result.append(l[1:length[i]-1])
        return  result


    def train(self,train_datas,test_datas):


        train_ids, train_input_ids, train_input_masks, train_segment_ids, train_labels, train_words = self.createInPut(train_datas, FLAGS.max_seq_length)


        num_train_steps = int(  # 这个是总的迭代次数，训练完一轮的次数*总的次数
            len(train_ids) / FLAGS.batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)  # 总的迭代次数 * 0.1

        test_ids,test_input_ids,test_input_masks,test_segment_ids,test_labels,test_words=self.createInPut(test_datas,FLAGS.max_seq_length)

        self.train_op = optimization.create_optimizer(
            self.total_loss, FLAGS.learning_rate,num_train_steps, num_warmup_steps, False)



        e = evaluateUtils()
        loss=[]
        predict=[]
        init_op=tf.global_variables_initializer()# 创建完train_op后初始化
        saver = tf.train.Saver(max_to_keep=10)

        idx_to_tag = {idx: tag for tag, idx in FLAGS.label_map[0].items()}
        with tf.Session() as sess:
            sess.run(init_op)
            for i in range(num_train_steps):
                shuffIndex = np.random.permutation(np.arange(len(train_ids)))
                shuffIndex=shuffIndex[0:FLAGS.batch_size]
                feed = self.getFeed(train_input_ids[shuffIndex], train_input_masks[shuffIndex], train_segment_ids[shuffIndex], train_labels[shuffIndex])
                train_s_labels=train_labels[shuffIndex]
                _,los,pred,lengths=sess.run([self.train_op,self.total_loss,self.predict,self.lengths],feed_dict=feed)
                print(i)
                if i%400==0 and i>1  :#
                    # 这里测试一batch的train和全部的test
                    print("训练集合loss为:%.3f"%los)
                    # 这里是train的评估
                    pred = self.split(pred, lengths)
                    real_label = self.split(train_s_labels, lengths)
                    e.evaulateOflogits(pred, real_label, idx_to_tag,tokens=train_words[shuffIndex])
                    loss.clear()
                    predict.clear()
                    self.evaulateTest(sess, test_ids, test_input_ids, test_input_masks, test_segment_ids, test_labels,test_words,e,idx_to_tag)
                    self.evaulateTest(sess, train_ids, train_input_ids, train_input_masks, train_segment_ids, train_labels,train_words,e,idx_to_tag)
            #             saver.save(sess, os.path.join(FLAGS.log_root_path, "bert_ner"), global_step=i)



    def getFeed(self,input_ids,input_masks,segment_ids,labels=None):

           if labels is None:
               feed = {self.input_ids: input_ids,
                       self.input_mask: input_masks, self.segment_ids: segment_ids
                       }

           else:
               feed={self.input_ids:input_ids,
                     self.input_mask:input_masks,self.segment_ids:segment_ids,
                     self.input_labels:labels}
           return feed






    def createModel(self,input_ids,input_mask,segment_ids,labels,is_training):
        if FLAGS.max_seq_length > self.bert_config.max_position_embeddings: # 模型有个最大的输入长度 512
            raise ValueError("超出模型最大长度")
            # 创建bert模型
        model = modeling.BertModel(
            config=self.bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False # 这里如果使用TPU 设置为True，速度会快些。使用CPU 或GPU 设置为False ，速度会快些。
        )
        self.embedding = model.get_sequence_output()# 这个获取每个token的output 输入数据[batch_size, seq_length, embedding_size] 如果做seq2seq 或者ner 用这个
        hidden_size = self.embedding.shape[-1].value #获取输出的维度
        max_seq_length = self.embedding.shape[1].value# 获取句子长度

        used = tf.sign(tf.abs(input_ids))
        self.lengths = tf.reduce_sum(used, reduction_indices=1)  # [batch_size] 大小的向量，包含了当前batch中的序列长度

        blstm_crf = BLSTM(embedded_chars=self.embedding, hidden_unit=FLAGS.lstm_size, cell_type=FLAGS.cell,
                              num_layers=FLAGS.num_layers,
                              droupout_rate=FLAGS.droupout_rate, initializers=initializers, num_labels=FLAGS.num_labels,
                              seq_length=max_seq_length, labels=labels, lengths=self.lengths, is_training=is_training)
        blstm_crf.add_blstm_crf_layer()
        self.total_loss =blstm_crf.totle_loss
        self.predict = blstm_crf.predict
        self.trans=blstm_crf.trans




    def loadModel(self,init_checkpoint):
        tvars = tf.trainable_variables()
        # 加载模型，如果init_checkpoint 只有bert 在训练的时候还需要初始化下
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.logging.info("**** Trainable Variables ****")
        # 打印加载模型的参数
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)


    def predict_f(self,sess,feed):
        los, pred, lengths = sess.run([self.total_loss, self.predict, self.lengths], feed_dict=feed)
        return los,pred,lengths

    def predictForInputData(self,inputData):# 这里input 是个list

        ids, input_ids, input_masks, segment_ids,labels,words=self.createInPut(inputData, FLAGS.max_seq_length)
        feed=self.getFeed(input_ids, input_masks, segment_ids)
        return  self.sess.run([self.predict, self.lengths], feed_dict=feed)


    def loadModelForPredict(self,path):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
        sess = tf.Session(config=config)
        saver = tf.train.Saver()
        saver.restore(sess, path)
        self.saver=saver
        self.sess=sess

    # 将label 合并。

    def minibatchesNdArray(self,test_ids, test_input_ids, test_input_masks, test_segment_ids, test_labels, words,
                           minibatch_size):
        iterNum = len(test_ids) // minibatch_size
        for i in range(iterNum):
            start = i * minibatch_size
            end = (i + 1) * minibatch_size
            yield test_ids[start:end], test_input_ids[start:end], test_input_masks[start:end], test_segment_ids[start:end], test_labels[start:end], words[start:end]

    def evaulateTest(self,sess,test_ids,test_input_ids,test_input_masks,test_segment_ids,test_labels,words,e,idx_to_tag,combine_type_list=None):
        loss=[]
        predict=[]
        real_labels=[]
        tokens=[]
        total_lengths=[]
        for test_ids_batch, test_input_ids_batch, test_input_masks_batch, test_segment_ids_batch, test_labels_batch,test_words in self.minibatchesNdArray(
                test_ids, test_input_ids, test_input_masks, test_segment_ids, test_labels,words, FLAGS.batch_size):
            feed = self.getFeed(test_input_ids_batch, test_input_masks_batch, test_segment_ids_batch, test_labels_batch)
            los, pred, lengths = self.predict_f(sess,feed)
            loss.append(los)
            tokens.extend(test_words.tolist())
            predict.extend(pred.tolist())
            real_labels.extend(test_labels_batch.tolist())
            total_lengths.extend(lengths.tolist())
        # 这里打印测试的信息
        print("测试集合loss为:%.3f" % np.mean(loss))
        # 这里是train的评估
        pred = self.split(predict, total_lengths)
        real_label = self.split(real_labels, total_lengths)
        e.evaulateOflogits(pred, real_label, idx_to_tag,combine_type_list,tokens,True)



    def convert_single_example( self,tokens,labels, max_seq_length):
        """
        将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中
        :param example: 一个样本
        :param label_list: 标签列表
        :param max_seq_length:
        :param tokenizer:
        :param mode:
        :return:
        """
        # tokens=list(itertools.chain(*list( #
        #     map(lambda x: ["[UNK]"] if len(self.tokenizer.tokenize(x)) > 1 else self.tokenizer.tokenize(x), tokens))))
        tokens=list(itertools.chain(*list( #  todo 这里将来需要和标签同步
            map(lambda x: ["[UNK]"] if len(self.tokenizer.tokenize(x)) != 1 else self.tokenizer.tokenize(x), tokens))))

        # if labels
        assert len(tokens)==len(labels)

        # 序列截断
        if len(tokens) >= max_seq_length - 1:       #最后 加上句首句尾 长度为128
            tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
            labels = labels[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")  # 句子开始设置CLS 标志
        segment_ids.append(0)
        # append("O") or append("[CLS]") not sure!
        label_ids.append(FLAGS.label_map[0]["[CLS]"])  # 句首和句尾使用不同的标志来标注
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(FLAGS.label_map[0][labels[i]])


        ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
        segment_ids.append(0)
        # append("O") or append("[SEP]") not sure!
        label_ids.append(FLAGS.label_map[0]["[SEP]"])
        input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)  # 将序列中的字(ntokens)转化为ID形式
        input_mask = [1] * len(input_ids)
        # padding, 使用
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            # we don't concerned about it!
            label_ids.append(FLAGS.label_map[0]["[PAD]"])
            ntokens.append("**NULL**")
            # label_mask.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        return input_ids,input_mask,segment_ids,label_ids,ntokens


    # 这里输入input对象，转换成 要feed的对象，后面还会封装一个feed的方法。方便训练预测，验证时候调用
    def createInPut(self,inputDatas,max_seq_length):
        input_idsArray=[]
        input_maskArray=[]
        segment_idsArray=[]
        label_idsArray=[]
        ntokensArray=[]
        sample_idsArray=[]
        for input in  inputDatas:
            texts = input[1].split(" ")
            labels = input[0].split(" ")
            sample_id = 0 # 这里可以设置样本ID 在后面分析的时候，可以方便找到原文。
            input_ids, input_mask, segment_ids, label_ids, ntokens=self.convert_single_example(texts,labels, max_seq_length) # 这个过程太慢，需要优化
            sample_idsArray.append(sample_id)
            input_idsArray.append(input_ids)
            input_maskArray.append(input_mask)
            segment_idsArray.append(segment_ids)
            label_idsArray.append(label_ids)
            ntokensArray.append(ntokens)
        return np.asarray(sample_idsArray,dtype=np.int32),np.asarray(input_idsArray,dtype=np.int32),np.asarray(input_maskArray,dtype=np.int32),\
               np.asarray(segment_idsArray, dtype=np.int32),np.asarray(label_idsArray, dtype=np.int32),np.asarray(ntokensArray,dtype=np.str)





