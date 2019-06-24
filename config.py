import tensorflow as tf



import os


flags = tf.flags

FLAGS = flags.FLAGS

if os.name == 'nt':
    bert_path = 'E:/bert/chinese_L-12_H-768_A-12'
    root_path = os.path.dirname(os.path.realpath(__file__))
    log_root_path=""
else:
    bert_path = '/home/bert/chinese_L-12_H-768_A-12'
    root_path = os.path.dirname(os.path.realpath(__file__))
    log_root_path="" # 保存模型的地方

flags.DEFINE_string(
    "data_dir", os.path.join(root_path, 'NERdata'),
    "The input datadir.",
)

flags.DEFINE_string('log_root_path',log_root_path , 'the path of save model')
flags.DEFINE_string(
    "bert_config_file", os.path.join(bert_path, 'bert_config.json'),
    "The config json file corresponding to the pre-trained BERT model."
)

flags.DEFINE_string(
    "task_name", 'ner', "The name of the task to train."
)



## Other parameters
flags.DEFINE_string(
    "init_checkpoint", os.path.join(bert_path, 'bert_model.ckpt'),
    "Initial checkpoint (usually from a pre-trained BERT model)."
)

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text."
)

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization."
)

flags.DEFINE_boolean('clean', True, 'remove the files which created by last training')

flags.DEFINE_bool("do_train", True, "Whether to run training."
)
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", True, "Whether to run the model in inference mode on the test set.")



flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 100.0, "Total number of training epochs to perform.")
flags.DEFINE_float('droupout_rate', 0.5, 'Dropout rate')
flags.DEFINE_float('clip', 5, 'Gradient clip')
flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_string("vocab_file", os.path.join(bert_path, 'vocab.txt'),
                    "The vocabulary file that the BERT model was trained on.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")
flags.DEFINE_string('data_config_path', os.path.join(root_path, 'data.conf'),
                    'data config file, which save train and dev config')
# lstm parame
flags.DEFINE_integer('lstm_size', 256, 'size of lstm units')
flags.DEFINE_integer('num_layers', 1, 'number of rnn layers, default is 1')

flags.DEFINE_integer("batch_size",40,"number of batch")


flags.DEFINE_string('cell', 'lstm', 'which rnn cell used')



flags.DEFINE_list("label_list",['O', 'B-ORG', 'B-LOC', 'I-PER', 'I-ORG', 'I-LOC', 'B-PER',"[CLS]","[SEP]","[PAD]"],"label list") #CLS SEP PAD 对应 bert中的 CLS SEP PAD


label_map = {}

for (i, label) in enumerate(FLAGS.label_list):
    label_map[label] = i
flags.DEFINE_list("label_map",[label_map],"label_map list") # 因为flag不用定义字典，就用list代替

flags.DEFINE_integer('num_labels', len(label_map), 'number of rnn labels')
