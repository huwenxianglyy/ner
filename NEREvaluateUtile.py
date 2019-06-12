import numpy as np
import re
import os



save_error_path="/home/huwenxiang/deeplearn/court_data/result/"

# 这里建议other 的标签就是O
class evaluateModel:
    def __init__(self):
        self.realNum=0
        self.predictNum=0
        self.rightNum=0
        self.typeName=""
        self.errorSmple=[]

class evaluateUtils:


    def get_accu(self, real_labels, pre_labels):
        real_labels = np.asarray(real_labels)
        pre_labels = np.asarray(pre_labels)
        compare_result = real_labels - pre_labels
        right_indexs = np.where(compare_result == 0)[0]
        right_nums = np.shape(right_indexs)[0]
        m = np.shape(real_labels)[0]
        accu = right_nums / m
        return accu, right_nums, m


    #这个只是单纯的评测每个标签是否一致
    def ner_accu(self, real_labels, pre_labels, sequences_length):
        m = np.shape(sequences_length)[0]
        real_labels_list = list()
        pre_labels_list = list()
        for i in range(0, m):
            real_labels_list.extend(real_labels[i][:sequences_length[i]])
            pre_labels_list.extend(pre_labels[i][:sequences_length[i]])
        accu, right_nums, m = self.get_accu(real_labels_list, pre_labels_list, False)
        return accu, right_nums, m


    #
    def get_chunk_type(self,tok, idx_to_tag):
        """
        Args:
            tok: id of token, ex 4
            idx_to_tag: dictionary {4: "B-PER", ...}

        Returns:
            tuple: "B", "PER"

        """
        tag_name = idx_to_tag[tok]
        tag_class = tag_name.split('-')[0]
        tag_type = tag_name.split('-')[-1]
        return tag_class, tag_type




    # 把预测的序列，变成可读的tag  tag 的格式{"B-PER": 4, "I-PER": 5, "B-LOC": 3,"0-0":0}
    # IOB
    def get_chunks_IOB(self,seq, idx_to_tag):
        """Given a sequence of tags, group entities and their position

        Args:
            seq: [4, 4, 0, 0, ...] sequence of labels
            tags: dict["O"] = 4

        Returns:
            list of (chunk_type, chunk_start, chunk_end)

        Example:
            seq = [4, 5, 0, 3]
            tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
            result = [("PER", 0, 2),("0",2,3),("LOC", 3, 4)]

        """
        if len(seq)==0:
            return []

        chunks=[]
        #创建第一个桶
        bucket=[]
        tokZero=seq[0];
        tok_chunk_class, tok_chunk_type = self.get_chunk_type(tokZero, idx_to_tag)
        # 这里如过单独出现了I 就说名是错误的，当作其他来处理
        if tok_chunk_class == "I":
            tok_chunk_type = "O"
        bucket.append(tok_chunk_type)
        bucket.append(0)
        for i,tok in enumerate(seq):
            if i==0:continue
            tok_chunk_class, tok_chunk_type = self.get_chunk_type(tok, idx_to_tag)
            if self.isNeedEndBucket(bucket,tok_chunk_class,tok_chunk_type):
                #当前bucket的结束位置
                bucket.append(i)
                #加入队列
                chunks.append(bucket)
                #初始化捅
                bucket=[]
                #这里如过单独出现了I 就说名是错误的，当作其他来处理
                if tok_chunk_class=="I":
                    tok_chunk_type="O"
                bucket.append(tok_chunk_type)
                bucket.append(i)

        bucket.append(i+1)
        chunks.append(bucket)


        return chunks

    def isNeedEndBucket(self,bucket,tok_chunk_class,tok_chunk_typ):

        if bucket[0]==tok_chunk_typ and tok_chunk_class=="I":return  False

        return  True




  # 把预测的序列，变成可读的tag  tag 的格式{"B-PER": 4, "I-PER": 5, "E-PRE":6,"B-LOC": 3,"0-0":0}
    # EIOB
    def get_chunks_EIOB(self,seq, tags):
        '''
        结构和上面的一致，只不过是多了个End标识符
        '''
        if len(seq)==0:
            return []
        idx_to_tag = {idx: tag for tag, idx in tags.items()}
        chunks=[]
        #创建第一个桶
        bucket=[]
        tokZero=seq[0];
        tok_chunk_class, tok_chunk_type = self.get_chunk_type(tokZero, idx_to_tag)
        if tok_chunk_class == "I" or tok_chunk_class=="E":
            tok_chunk_type = "0"
        bucket.append(tok_chunk_type)
        bucket.append(0)
        last_tok_chunk_class=None
        for i,tok in enumerate(seq):
            if i == 0:continue
            tok_chunk_class, tok_chunk_type = self.get_chunk_type(tok, idx_to_tag)
            if self.isNeedEndBucket_EIOB(bucket,tok_chunk_class,tok_chunk_type,last_tok_chunk_class):


                #这里要判断之前的是否合乎规则
                if last_tok_chunk_class=="I":
                    start=bucket[1]
                    for j in range(start,i):
                        bucket=[]
                        bucket.append("0")
                        bucket.append(j)
                        bucket.append(j+1)
                        chunks.append(bucket)
                else:
                    # 当前bucket的结束位置
                    bucket.append(i)
                    #加入队列
                    chunks.append(bucket)
                #初始化捅
                bucket=[]
                #这里如过单独出现了I 就说名是错误的，当作其他来处理
                if tok_chunk_class=="I" or tok_chunk_class=="E":
                    tok_chunk_type="0"
                bucket.append(tok_chunk_type)
                bucket.append(i)
                #
            last_tok_chunk_class=tok_chunk_class
        #最后一个加入到数组中 todo 如果最后一个不符合规则，要处理下
        bucket.append(i+1)
        chunks.append(bucket)
        return chunks

    #如果上个标签是E 就结束，如果是I或E并且类型和之前的一致就继续。
    def isNeedEndBucket_EIOB(self,bucket,tok_chunk_class,tok_chunk_typ,last_tok_chunk_class):
        if last_tok_chunk_class=="E":return  True
        if bucket[0]==tok_chunk_typ and (tok_chunk_class=="I" or tok_chunk_class=="E"):return  False
        return  True



    def evaluate(self,perdict,label,r,p,tags,tokens=None,is_output=False):
        '''
        评估模型的好坏。输入是，[["PER", 0, 2],["0",2,3],["LOC", 3, 4]]
        '''
        statisticMap={}
        self.statisticPredictInfo(statisticMap,perdict,label,r,p,tags,tokens,is_output)
        self.statisticRightNum(statisticMap,label)
        self.perrty(statisticMap)



    # 统计每个类别预测了多少个，正确了多少
    def statisticPredictInfo(self,statisticMap,perdict,label,r,p,tags,tokens,is_output=False):
        for index, line in enumerate(perdict):
            real_l = label[index]
            token=tokens[index] if tokens is not None else None

            for l in line:
                typeName = l[0]
                evaluateM = self.getEvaluateModel(statisticMap, typeName)
                evaluateM.predictNum += 1
                if l in real_l:
                    evaluateM.rightNum += 1
                else:# 这里将错误的样本加入后期可以打印出来看看
                    #获取错误的标签，保存到对应的文件
                    type=l[0]
                    if is_output :# 如果需要输出，会把错误的seq输出，具体分析。
                        real = r[index]
                        predict = p[index]
                        token = tokens[index][1:len(real) + 1]
                        self.writeFile2(os.path.join(save_error_path,type)+".txt",
                                        list(map(lambda x: tags[x], real)), list(map(lambda x: tags[x], predict)),
                                        token)
                    evaluateM.errorSmple.append((token,real_l,line))

    def writeFile(self,path,answer,predict,text):
        try:  # 这里写入不准确的
            with open(path, mode="a+", encoding="utf-8") as f1:
                f1.write("原文:%s"%text+"\n")
                f1.write("答案:%s" % (answer) + "\n")
                f1.write("预测:%s" % (predict) + "\n")
                f1.write("\n")
        except:
            print("写入结果异常")



    def writeFile2(self,path,answer,predict,text):# 这个是真正ner 写入方式
        try:
            with open(path, mode="a+", encoding="utf-8") as f1:
                for t,a,p in zip(text,answer,predict):
                    f1.write("%s\t%s\t%s"%(t,a,p))
                    f1.write("\n")
                f1.write("。\n")
                f1.write("\n")
        except:
            print("写入结果异常")

    def reCoverPunction(self,token,line):# 这里是吧标点还原
        words=token[1:-1]
        result=[]
        for l in line:
            type,start,end=l
            result.append("".join(words[start:end]))
        result=",".join(result)
        return result

    def combine_obj(self,labels, combain_labels_list):  # ORG,PLACE
        combine_labels = labels[::-1]  # 这里先翻转。
        after_label = "O"  # 这里的after 和before是相对于翻转前的顺序。
        for i, label in enumerate(combine_labels):
            label_split = label.split("-")
            BItype, type = label_split[0], label_split[-1]
            currentBItype = BItype  # 当前的BI
            currentType = type  # 当前的TYPE
            if type in combain_labels_list:  # 当前标签在 需要处理的list中
                # 这里先确定头，看前一个是什么标签，如果也在 需要处理的list中 ，那么当前BI 就I 否则就 B
                before_label = "O" if i + 1 == len(combine_labels) else combine_labels[i + 1].split("-")[-1]
                if before_label in combain_labels_list:  #
                    currentBItype = "I"
                else:
                    currentBItype = "B"
                if after_label in combain_labels_list:
                    currentType = after_label
            combine_labels[i] = currentType if currentType == currentBItype else currentBItype + "-" + currentType
            after_label = currentType
        return combine_labels[::-1]

    #打印信息。
    def perrty(self,statisticMap):
        errorList=[]
        print("类别\t\t准确率\t\t召回率\t\tF1SCORE\t\t类别数量")
        ite=sorted(statisticMap.items(),key=lambda x:x[0])
        for key,value in ite:
            if len(value.errorSmple) >0:
                errorList.append(value.errorSmple[0])
            rightNum=value.rightNum
            realNum =value.realNum
            predictNum =value.predictNum
            acc=rightNum/predictNum if predictNum!=0 else 0
            recell=rightNum/realNum if realNum!=0 else 0
            f1=2*acc*recell/(acc+recell+1e-10)
            print("%s\t\t%.3f\t\t%.3f\t\t%.3f\t\t%d"%(key,acc,recell,f1,realNum))
        for error in errorList:# 这里打印一个错误的看看
            print("原文:%s"%error[0])
            print("答案:%s"%error[1])
            print("预测:%s"%error[2])



    #统计标准答案中没个类别数量
    def statisticRightNum(self,statisticMap,label):
        for line in label:
            for l in line:
                typeName = l[0]
                evaluateM=self.getEvaluateModel(statisticMap,typeName)
                evaluateM.realNum+=1



    def getEvaluateModel(self,statisticMap,typeName):
        if typeName in statisticMap:
            return  statisticMap[typeName]
        evaluateM=evaluateModel()
        evaluateM.typeName=typeName
        statisticMap[typeName]=evaluateM
        return evaluateM

    def evaulateOflogits(self,predicts,reals,tags,combine_type_list=None,tokens=None,is_output=False):
        real_result=[]
        predict_result=[]
        tag_to_id = {tag: idx for idx, tag in tags.items()}
        for i,real_predict in enumerate(zip(reals, predicts)):
            real, predict=real_predict
            # 这里加入是否合并的控制。如果合并就把ID转换成B-的格式，然后在转换成 id
            if combine_type_list is not None:
                real_BI=[tags[index] for index in real]
                predict_BI=[tags[index] for index in predict]
                real_BI=self.combine_obj(real_BI,combine_type_list)
                predict_BI=self.combine_obj(predict_BI,combine_type_list)
                real=[tag_to_id[t] for t in real_BI]
                predict=[tag_to_id[t] for t in predict_BI]
                reals[i]=real # 这里要写入回去。
                predicts[i]=predict
            real_IOB = self.get_chunks_IOB(real, tags)
            # result = evaluateUtils.get_chunks_EIOB(line,tags)
            predict_IOB = self.get_chunks_IOB(predict, tags)
            real_result.append(real_IOB)
            predict_result.append(predict_IOB)
        self.evaluate(predict_result,real_result,reals,predicts,tags,tokens,is_output)



