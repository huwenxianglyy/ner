import utils
import bert_ner_model
from  config import FLAGS







model=bert_ner_model.ner_model(True)
model.loadModel(FLAGS.init_checkpoint)#

test_data=utils.read_data("./NERdata/test.txt")
train_data=utils.read_data("./NERdata/train.txt")





model.train(train_data,test_data)

