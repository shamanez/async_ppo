"""
State-Value Function

Written by Patrick Coady (pat-coady.github.io)

Modified by Tin-Yin Lai (wu6u3) into asynchronous version
"""

import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
#import os

class NNValueFunction(object):
    """ NN-based state-value function """
    def __init__(self, obs_dim, hid1_mult, thread_idx, shared_nn):
        """
        Args:
            obs_dim: number of dimensions in observation vector (int)
            hid1_mult: size of first hidden layer, multiplier of obs_dim
        """
        self.replay_buffer_x = None
        self.replay_buffer_y = None
        self.obs_dim = obs_dim
        self.hid1_mult = hid1_mult
        self.epochs = 10
        self.lr = None  # learning rate set in _build_graph()
        self._thread_idx=thread_idx # -1 for global
        self._scope_name = "nn_net_"+str(self._thread_idx)
        self._build_graph()
        #self.sess = tf.Session(graph=self.g)
        #self.sess.run(self.init)

        var_refs = [v._ref() for v in self.get_vars()]
        self.gradients = tf.gradients(
            self.loss, var_refs,
            gate_gradients=False,
            aggregation_method=None,
            colocate_gradients_with_ops=False)
        self.apply_gradients=None
        self.sync = self.sync_from(shared_nn)

        #self. global_fit = self.fit_for_global(x=None, y=None, logger=None)

    def _build_graph(self):
        """ Construct TensorFlow graph, including loss function, init op and train op """
        with tf.variable_scope(self._scope_name) as scope:
            self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs_valfunc')
            self.val_ph = tf.placeholder(tf.float32, (None,), 'val_valfunc')
            # hid1 layer size is 10x obs_dim, hid3 size is 10, and hid2 is geometric mean
            hid1_size = self.obs_dim * self.hid1_mult  # default multipler 10 chosen empirically on 'Hopper-v1'
            hid3_size = 5  # 5 chosen empirically on 'Hopper-v1'
            hid2_size = int(np.sqrt(hid1_size * hid3_size))
            # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
            self.lr = 1e-2 / np.sqrt(hid2_size)  # 1e-3 empirically determined
            print('Value Params -- h1: {}, h2: {}, h3: {}, lr: {:.3g}'
                  .format(hid1_size, hid2_size, hid3_size, self.lr))
            # 3 hidden layers with tanh activations
            out = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / self.obs_dim)), name="h1")

            out = tf.layers.dense(out, hid2_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / hid1_size)), name="h2")
            out = tf.layers.dense(out, hid3_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / hid2_size)), name="h3")
            out = tf.layers.dense(out, 1,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / hid3_size)), name='output')
            self.out = tf.squeeze(out)
            self.loss = tf.reduce_mean(tf.square(self.out - self.val_ph))  # squared loss
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.loss)

            #self.init = tf.global_variables_initializer()
            self.h1_w, self.h1_b = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self._scope_name+'/h1')
            self.h2_w, self.h2_b = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self._scope_name+'/h2')
            self.h3_w, self.h3_b = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self._scope_name+'/h3')
            self.output_w, self.output_b =tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self._scope_name+'/output')

            scope.reuse_variables()
           
               
        #self.sess = tf.Session(graph=self.g)
        #self.sess.run(self.init)
    def fit_for_global(self, x, y, logger):
        """ Fit model to current data batch + previous data batch

        Args:
            x: features
            y: target
            logger: logger to save training loss and % explained variance
        """
        num_batches = max(x.shape[0] // 256, 1)
        batch_size = x.shape[0] // num_batches
        y_hat = self.predict(sess, x)  # check explained variance prior to update
        old_exp_var = 1 - np.var(y - y_hat)/np.var(y)
        if self.replay_buffer_x is None:
            x_train, y_train = x, y
        else:
            x_train = np.concatenate([x, self.replay_buffer_x])
            y_train = np.concatenate([y, self.replay_buffer_y])
        self.replay_buffer_x = x
        self.replay_buffer_y = y
        for e in range(self.epochs):
            x_train, y_train = shuffle(x_train, y_train)
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                feed_dict = {self.obs_ph: x_train[start:end, :],
                             self.val_ph: y_train[start:end]}
                _, l = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        y_hat = self.predict(sess, x)
        loss = np.mean(np.square(y_hat - y))         # explained variance after update
        exp_var = 1 - np.var(y - y_hat) / np.var(y)  # diagnose over-fitting of val func

        logger.log({'ValFuncLoss': loss,
                    'ExplainedVarNew': exp_var,
                    'ExplainedVarOld': old_exp_var})

    def fit(self, sess, x, y, logger):
        """ Fit model to current data batch + previous data batch

        Args:
            x: features
            y: target
            logger: logger to save training loss and % explained variance
        """
        num_batches = max(x.shape[0] // 256, 1)
        batch_size = x.shape[0] // num_batches
        y_hat = self.predict(sess, x)  # check explained variance prior to update
        old_exp_var = 1 - np.var(y - y_hat)/np.var(y)
        if self.replay_buffer_x is None:
            x_train, y_train = x, y
        else:
            x_train = np.concatenate([x, self.replay_buffer_x])
            y_train = np.concatenate([y, self.replay_buffer_y])
        self.replay_buffer_x = x
        self.replay_buffer_y = y
        for e in range(self.epochs):
            x_train, y_train = shuffle(x_train, y_train)
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                feed_dict = {self.obs_ph: x_train[start:end, :],
                             self.val_ph: y_train[start:end]}
                _, l = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        y_hat = self.predict(sess, x)
        loss = np.mean(np.square(y_hat - y))         # explained variance after update
        exp_var = 1 - np.var(y - y_hat) / np.var(y)  # diagnose over-fitting of val func

        logger.log({'ValFuncLoss': loss,
                    'ExplainedVarNew': exp_var,
                    'ExplainedVarOld': old_exp_var})

    def predict(self, sess, x):
        """ Predict method """
        feed_dict = {self.obs_ph: x}
        y_hat = sess.run(self.out, feed_dict=feed_dict)

        return np.squeeze(y_hat)

    #def close_sess(self):
    #    """ Close TensorFlow session """
    #    sess.close()
    def get_vars(self):
        return [self.h1_w, self.h1_b,
                self.h2_w, self.h2_b,
                self.h3_w, self.h3_b,
                self.output_w, self.output_b ]

#        weights = []

        #name = []
        #for tensor in self.g.as_graph_def().node:
        #    name.append(tensor.name)
        #print(name)

        #with self.g.as_default() as g:
#        with tf.variable_scope(self._scope_name) as scope:
#            weights.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope))
            # weights.append(g.get_tensor_by_name('h1/kernel:0'))
            # weights.append(g.get_tensor_by_name('h1/bias:0'))
            # weights.append(g.get_tensor_by_name('h2/kernel:0'))
            # weights.append(g.get_tensor_by_name('h2/bias:0'))
            # weights.append(g.get_tensor_by_name('h3/kernel:0'))
            # weights.append(g.get_tensor_by_name('h3/bias:0'))
  


#        return weights

    def sync_from(self, shared_nn, name=None):
        if shared_nn != None:
            src_vars = shared_nn.get_vars()
            dst_vars = self.get_vars()
            sync_ops = []

            with tf.name_scope(name, self._scope_name, []) as name:
        
                for(src_var, dst_var) in zip(src_vars, dst_vars):
  
                    sync_op = tf.assign(dst_var, src_var)
                    sync_ops.append(sync_op)
        
            return tf.group(*sync_ops, name=name)
        else:
            return None


