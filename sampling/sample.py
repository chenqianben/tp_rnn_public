from __future__ import print_function
import tensorflow as tf

import argparse
import os
from six.moves import cPickle
import codecs

from sampling.model import Model

from six import text_type


def sample(save_dir='save', n=1000, prime=u' ', sample=1):
    tf.reset_default_graph()
    tf.set_random_seed(100)
    with open(os.path.join(save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)
    model = Model(saved_args, training=False)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            sample_test = model.sample(sess, chars, vocab, n, prime,
                               sample)
            with codecs.open(os.path.join(save_dir, "sample.txt"), "w", "utf-8-sig") as temp:
    	        temp.write(sample_test)

            print(sample_test)

