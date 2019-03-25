import tensorflow as tf 
import numpy as np
import time
import os
import datetime

import modules as model
from options import Options
from utils import Dataset

class DoseModel(object):
    def __init__(self, opts, is_training=False):
        self.h = opts.image_size_h
        self.w = opts.image_size_w
        self.c = opts.channels
        self.opts = opts
        self.true_images = tf.placeholder(tf.float32, [None, self.c, self.h, self.w], "images")
        self.input_ptv = tf.placeholder(tf.float32, [None, self.opts.code_len], "input_ptv")
        self.input_oct = tf.placeholder(tf.float32, [None, self.opts.code_len], "input_ptv")
        self.lr = tf.placeholder(tf.float32, [], "learning_rate")
        self.is_training = self.opts.train
        self.predicted_imgs = self.create_network()
        self.loss = self.loss()

        self.sess = tf.Session()
        with tf.variable_scope("Optimizers"):
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss, var_list = tf.trainable_variables())
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef)
        tf.summary.scalar('Loss: ', self.loss)
        tf.summary.scalar('Learning Rate', self.lr)
        tf.summary.image('Predicted image', self.predicted_imgs, max_outputs=4)
        self.summaries = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.opts.root_dir+self.opts.summary_dir, self.sess.graph)

    def create_network(self):

        ### PTV Network Path ###
        ptv_conv1 = tf.layers.conv2d(self.input_ptv, filters=64, kernel_size=1, use_bias=False, name="init_conv1", data_format='channels_first')
        ptv_conv2 = tf.layers.conv2d(ptv_conv1, filters=64, kernel_size=7, strides=2, padding='same', use_bias=False, name="init_conv2", data_format='channels_first')
        ptv_block1 = model.block_layer(ptv_conv2, 64, 3, 2, self.is_training, "ptv_block1", data_format='channels_first')
        ptv_block2 = model.block_layer(ptv_block1, 128, 4, 2, self.is_training, "ptv_block2", data_format='channels_first')
        ptv_block3 = model.block_layer(ptv_block2, 256, 6, 2, self.is_training, "ptv_block3", data_format='channels_first')
        ptv_block4 = model.block_layer(ptv_block3, 512, 3, 2, self.is_training, "ptv_block4", data_format='channels_first')

        ### OAR-CT Network Path ###
        oct_conv1 = tf.layers.conv2d(self.input_oct, filters=64, kernel_size=1, use_bias=False, data_format='channels_first')
        oct_conv2 = tf.layers.conv2d(oct_conv1, filters=64, kernel_size=7, strides=2, padding='same', use_bias=False, data_format='channels_first')
        oct_block1 = model.block_layer(oct_conv2, 64, 3, 2, self.is_training, "oct_block1", data_format='channels_first')
        oct_block2 = model.block_layer(oct_block1, 128, 4, 2, self.is_training, "oct_block2", data_format='channels_first')
        oct_block3 = model.block_layer(oct_block2, 256, 6, 2, self.is_training, "oct_block3", data_format='channels_first')
        oct_block4 = model.block_layer(oct_block3, 512, 3, 2, self.is_training, "oct_block4", data_format='channels_first')

        embed = tf.add(ptv_block4, oct_block4)

        ### Anti-ResNet Network Path ###
        antires_block1 = model.block_layer(embed, 1024, 3, 2, self.is_training, "antires_block1", data_format='channels_first', transpose=True)
        antires_block1 = ptv_block3 + oct_block3 + antires_block1# skip-connection 1
        antires_block2 = model.block_layer(antires_block1, 512, 6, 2, self.is_training, "antires_block2", data_format='channels_first', transpose=True)
        antires_block2 = ptv_block2 + oct_block2 + antires_block2 # skip-connection 2
        antires_block3 = model.block_layer(antires_block2, 256, 4, 2, self.is_training, "antires_block3", data_format='channels_first', transpose=True)
        antires_block3 = ptv_block1 + oct_block1 + antires_block3# skip-connection 3
        antires_block4 = model.block_layer(antires_block3, 128, 3, 2, self.is_training, "antires_block4", data_format='channels_first', transpose=True)     
        antires_block4 = ptv_conv2 + oct_conv2 + antires_block4 # skip-connection 4
        antires_block5 = model.block_layer(antires_block4, 64, 1, 2, self.is_training, "antires_block5", data_format='channels_first', transpose=True)  
        antires_endconv = tf.layers.conv2d(antires_block5, filters=3, kernel_size=1, use_bias=False, name="final_conv", data_format='channels_first') 

        return antires_endconv

    def loss(self):
        with tf.variable_scope("loss"):
            loss = tf.losses.mean_squared_error(self.true_images, self.predicted_imgs)
            return loss
    
    def train(self):
        lr = self.lr
        self.sess.run(self.init)
        train_set = Dataset(self.opts)
        # TODO: Get num_epochs from opts
        for epoch in range(1, 100):
            batch_num = 0
            for batch_begin, batch_end in zip(xrange(0, self.opts.train_size, self.opts.batch_size), \
				xrange(self.opts.batch_size, self.opts.train_size, self.opts.batch_size)):
                begin_time = time.time()
                input_ptv, input_oct, gt_img = train_set.load_batch(batch_begin, batch_end)
                feed_dict = {self.true_images:gt_img, self.input_ptv:input_ptv, self.lr:lr, self.input_oct:input_oct}
                _, loss, summary = self.sess.run([self.optimizer, self.loss, self.summaries], feed_dict=feed_dict)
                
                batch_num += 1
                self.writer.add_summary(summary, epoch * (self.opts.train_size/self.opts.batch_size) + batch_num)
                if batch_num % self.opts.display == 0:
                    rem_time = (time.time() - begin_time) * (self.opts.num_epochs-epoch) * (self.opts.train_size/self.opts.batch_size)
                    log  = '-'*20
                    log += ' Epoch: {}/{}|'.format(epoch, self.opts.num_epochs)
                    log += ' Batch Number: {}/{}|'.format(batch_num, self.opts.train_size/self.opts.batch_size)
                    log += ' Batch Time: {}\n'.format(time.time() - begin_time)
                    log += ' Remaining Time: {:0>8}\n'.format(datetime.timedelta(seconds=rem_time))
                    log += ' lr: {} loss: {}\n'.format(lr, loss)
                    print log
                if epoch % self.opts.lr_decay == 0 and batch_num == 1:
                    lr *= self.opts.lr_decay_factor
                if epoch % self.opts.ckpt_frq == 0 and batch_num == 1:
                    self.saver.save(self.sess, self.opts.root_dir+self.opts.ckpt_dir+"{}_{}_{}".format(epoch, lr, loss))
        raise NotImplementedError