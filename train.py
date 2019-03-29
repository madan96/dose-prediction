import io
import os
import sys
import time
import argparse
from time import gmtime, strftime
import tensorflow as tf

import nnet.architecture as network
import utils

import nnet

from options import Options

FLAGS = tf.app.flags.FLAGS

# # Data
tf.app.flags.DEFINE_string('root_dir', '/home/rishabh/work/dose-prediction', """Base Path""")
tf.app.flags.DEFINE_string('dataset_dir', 'train_data', """Path to data""")

# # Training
tf.app.flags.DEFINE_integer('batch_size', 4, """Batch size""")
tf.app.flags.DEFINE_integer('num_epochs', 100, """Max iterations for training""")
# tf.app.flags.DEFINE_integer('decay_after', 20, """Decay learning after iterations""")
tf.app.flags.DEFINE_integer('ckpt_frq', 10, """Frequency at which to checkpoint the model""")
# tf.app.flags.DEFINE_integer('train_size', 10000, """The total training size""")
tf.app.flags.DEFINE_integer('display', 2, """Display log of progress""")
# tf.app.flags.DEFINE_float('lr_decay', 0.9, """Learning rate decay factor""")
tf.app.flags.DEFINE_float('lr', 1e-4, """Learning rate""")
tf.app.flags.DEFINE_boolean('train', True, """Training or testing""")

# # Model Saving
tf.app.flags.DEFINE_string('ckpt_dir', "ckpt", """Checkpoint Directory""")
# tf.app.flags.DEFINE_string('sample_dir', "imgs", """Generate sample images""")
tf.app.flags.DEFINE_string('summary_dir', "summary", """Summaries directory""")

def main(_):
    print ('Beginning Run')

    net = network.DoseModel(FLAGS, True)

    print ('Training the network...')
    net.train()
    print ('Done training the network...\n')
        

if __name__ == '__main__':
    try:
        tf.app.run()
    except Exception as E:
        print (E)