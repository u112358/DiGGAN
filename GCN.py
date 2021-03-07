import os

import numpy as np
import tensorflow as tf

from utils import layers
from utils.data_reader import *
from utils import DirGen
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import scipy.io as sio
from scipy.misc import imsave
import sys

from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


def print_acc(acc):
    print('\t\t0\t\t30\t\t60\t\t90\t')
    print('0\t%lf\t%lf\t%lf\t%lf\t' % (acc[0, 0], acc[1, 0], acc[2, 0], acc[3, 0]))
    print('30\t%lf\t%lf\t%lf\t%lf\t' % (acc[0, 1], acc[1, 1], acc[2, 1], acc[3, 1]))
    print('60\t%lf\t%lf\t%lf\t%lf\t' % (acc[0, 2], acc[1, 2], acc[2, 2], acc[3, 2]))
    print('90\t%lf\t%lf\t%lf\t%lf\t' % (acc[0, 3], acc[1, 3], acc[2, 3], acc[3, 3]))


class GaitGAN:
    def __init__(self, tf_config):
        self.tf_config = tf_config
        self.GEI_height = 128
        self.GEI_width = 88
        self.GEI_channel = 1
        self.init_learning_rate = 1e-2
        self.beta1 = 0.5
        self.max_step = 100000
        self.batch_size = 30
        self.num_views = 14
        self.embeddings_dims = 52
        self.num_classes = 5154
        # build graph
        self.input = tf.placeholder(dtype=tf.float32,
                                    shape=(self.batch_size, self.GEI_height, self.GEI_width, self.GEI_channel),
                                    name='input')
        self.input_A = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.batch_size), name='input_A')
        self.labels = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.num_classes))
        # self.learning_rate = tf.placeholder(dtype=tf.float32, shape=1, name='learning_rate')
        # self.learning_rate = tf.placeholder(dtype=tf.float32, shape=1, name='learning_rate')

        # ops of the network
        # self.z = layers.encoder(self.input, n_layers=3, with_bn=[True, True, False], kernel_size=[7, 7, 7],
        #                         filters=[16, 64, 256], z_dim=self.embedding_dims, strides=[2, 2, 1])

        self.z = layers.encoder_GNN(self.input, self.input_A, z_dim=self.embeddings_dims)
        self.embeddings = tf.nn.l2_normalize(self.z, dim=1, epsilon=1e-12, name='embeddings')

        self.predictions = tf.layers.dense(self.embeddings, self.num_classes)
        self.cls_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predictions,
                                                                               labels=self.labels))
        self.graph_loss = layers.get_graph_reg_loss(
            self.embeddings, self.input_A)
        self.loss = self.cls_loss + self.graph_loss

        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.compat.v1.train.exponential_decay(self.init_learning_rate,
                                                                  self.global_step,
                                                                  10000, 0.5, staircase=True)

        self.train_op = tf.train.MomentumOptimizer(momentum=0.9, learning_rate=self.learning_rate).minimize(self.loss,
                                                                                                            global_step=self.global_step)
        # self.train_op = tf.train.MomentumOptimizer(momentum=0.9, learning_rate=self.learning_rate).minimize(self.loss)
        tf.summary.scalar('loss', self.loss, collections=['general'])
        self.sum_op = tf.summary.merge_all(key='general')

        self.result_input = tf.placeholder(dtype=tf.string, name='results')
        self.result_input_eer = tf.placeholder(dtype=tf.string, name='results_eer')
        self.acc_input = tf.placeholder(dtype=tf.float32, shape=4, name='acc')
        self.eer_input = tf.placeholder(dtype=tf.float32, shape=4, name='eer')

        self.sum_result = tf.summary.text('result', self.result_input, collections=['validate'])
        self.sum_result_eer = tf.summary.text('result_eer', self.result_input_eer, collections=['validate'])
        with tf.name_scope('acc_0'):
            self.sum_acc0 = tf.summary.scalar('acc_0_0', self.acc_input[0], collections=['validate'])
            self.sum_eer0 = tf.summary.scalar('eer_0_0', self.eer_input[0], collections=['validate'])
        with tf.name_scope('acc_30'):
            self.sum_acc0 = tf.summary.scalar('acc_30_30', self.acc_input[1], collections=['validate'])
            self.sum_acc0 = tf.summary.scalar('eer_30_30', self.eer_input[1], collections=['validate'])
        with tf.name_scope('acc_60'):
            self.sum_acc0 = tf.summary.scalar('acc_60_60', self.acc_input[2], collections=['validate'])
            self.sum_acc0 = tf.summary.scalar('eer_60_60', self.eer_input[2], collections=['validate'])
        with tf.name_scope('acc_90'):
            self.sum_acc0 = tf.summary.scalar('acc_90_90', self.acc_input[3], collections=['validate'])
            self.sum_acc0 = tf.summary.scalar('eer_90_90', self.eer_input[3], collections=['validate'])
        self.summary_op_general = tf.summary.merge_all(key='general')
        self.summary_op_validate = tf.summary.merge_all(key='validate')

    def train(self, k):
        with tf.Session(config=self.tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            script_name = sys.argv[0][:-3].split('/')[-1]
            logdir = DirGen.dir_gen('./exp/summary/' + script_name)
            modeldir = DirGen.dir_gen('./exp/model/')
            os.system('cp ' + script_name + '.py ' + logdir)
            with open(os.path.join(logdir, 'log.txt'), 'w') as file:
                file.write(script_name + '\n')
                file.write("learning rate:" + str(self.learning_rate) + '\n')
                file.close()
            self.summary_writer = tf.summary.FileWriter(graph=sess.graph, logdir=logdir)
            self.saver = tf.train.Saver(max_to_keep=200)
            # self.saver.restore(sess, './exp/model/20190902-194932/-7999')
            self.result_txt = 'Not updated yet'
            self.result_txt_eer = 'Not updated yet'
            self.acc = np.zeros((4, 4))
            self.eer = np.zeros((4, 4))

            dr = DataReader(root_path='./GEI/GEI', usage='Training')
            for step in tqdm(range(self.max_step)):
                x, y, p, A = dr.select_data(6, 5)
                # y = tf.one_hot(depth=5154, indices=y)
                # anchor, positive, labels = dr.select_pairs_for_contrastive_loss_v2(self.batch_size)
                # x = np.concatenate([np.asarray(anchor), np.asarray(positive)], 0)
                x = np.expand_dims(x, -1)
                # labels = np.expand_dims(y, -1)
                labels = np.eye(self.num_classes)[y]
                _, loss, sum = sess.run(
                    [self.train_op, self.loss, self.sum_op],
                    feed_dict={self.input: x, self.labels: labels, self.input_A: A})
                print('step:%d, loss:%lf' % (step, loss))
                self.summary_writer.add_summary(sum, step)

                if (step) % 2000 == 0:
                    if not os.path.exists(modeldir):
                        os.makedirs(modeldir)
                    self.saver.save(sess=sess, save_path=modeldir + '/', global_step=step)

                if (step) % 4000 == 0:
                    print('validating ...')
                    self.acc, self.result_txt = self.validate(dr, sess)
                    sum = sess.run(self.summary_op_validate, feed_dict={self.acc_input: self.form_acc_to_feed(),
                                                                        self.result_input: self.result_txt,
                                                                        self.eer_input: self.form_eer_to_feed(),
                                                                        self.result_input_eer: self.result_txt_eer})
                    self.summary_writer.add_summary(sum, step)
                    # acc_path = 'acc_%d.mat' % step
                    # sio.savemat(join(logdir, acc_path), {'acc': self.acc})

        del dr

    def validate(self, data_reader, sess):
        angles = [0, 30, 60, 90]
        acc = np.zeros((4, 4))
        with tqdm(total=16) as pbar:
            for i in range(4):
                for j in range(4):
                    acc[i, j] = self.validate_probe_n_at_gallery_m(data_reader, sess, angles[i], angles[j])
                    pbar.update(1)
        print_acc(acc)
        result_txt = self.plot_acc(acc)
        return acc, result_txt

    def validate_probe_n_at_gallery_m(self, data_reader, sess, n, m):
        list_probe_indices = data_reader.get_list_of_probe_at_angle_k(angle_k=n)
        len_of_list = list_probe_indices.shape[0]
        embeddings_probe = np.zeros((len_of_list, self.embeddings_dims))
        idx_probe = []
        nof_batch_probe = int(np.floor(len_of_list / self.batch_size))
        for i in range(nof_batch_probe):
            x, _idx_probe, _angle_probe = data_reader.get_probe_by_indices(
                list_probe_indices[i * self.batch_size:(i + 1) * self.batch_size])
            x = np.expand_dims(np.array(x), -1)
            A = np.eye(self.batch_size)
            emb = sess.run(self.embeddings, feed_dict={self.input: x, self.input_A: A})
            embeddings_probe[i * self.batch_size:(i + 1) * self.batch_size, :] = emb
            idx_probe.append(_idx_probe)

        list_gallery_indices = data_reader.get_list_of_gallery_at_angle_k(angle_k=m)
        len_of_list = list_gallery_indices.shape[0]
        embeddings_gallery = np.zeros((len_of_list, self.embeddings_dims))
        idx_gallery = []
        nof_batch_gallery = int(np.floor(len_of_list / self.batch_size))
        for i in range(nof_batch_gallery):
            x, _idx_gallery, _angle_gallery = data_reader.get_gallery_by_indices(
                list_gallery_indices[i * self.batch_size:(i + 1) * self.batch_size])
            x = np.expand_dims(np.array(x), -1)
            A = np.eye(self.batch_size)
            emb = sess.run(self.embeddings, feed_dict={self.input: x, self.input_A: A})
            embeddings_gallery[i * self.batch_size:(i + 1) * self.batch_size, :] = emb
            idx_gallery.append(_idx_gallery)

        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(embeddings_gallery[0:nof_batch_gallery * self.batch_size, :])
        distances, indices = nn.kneighbors(embeddings_probe[0:nof_batch_probe * self.batch_size, :])

        idx_gallery = np.reshape(np.asarray(idx_gallery), (-1, 1))
        idx_probe = np.reshape(np.asarray(idx_probe), (-1, 1))
        predicted = idx_gallery[indices, :]

        diff = np.reshape(predicted, (-1, 1)) - idx_probe[0:nof_batch_probe * self.batch_size, :]
        acc = np.where(diff == 0)[0].shape[0] / float(nof_batch_probe * self.batch_size)
        return acc

    def plot_acc(self, acc):
        tl = [
            ['', '0', '30', '60', '90'],
            ['0', str(acc[0, 0]), str(acc[1, 0]), str(acc[2, 0]), str(acc[3, 0])],
            ['30', str(acc[0, 1]), str(acc[1, 1]), str(acc[2, 1]), str(acc[3, 1])],
            ['60', str(acc[0, 2]), str(acc[1, 2]), str(acc[2, 2]), str(acc[3, 2])],
            ['90', str(acc[0, 3]), str(acc[1, 3]), str(acc[2, 3]), str(acc[3, 3])]
        ]
        # summary_acc_op = tf.summary.text('acc_table', tf.convert_to_tensor(tl))
        # text = sess.run(summary_acc_op)
        # self.summary_writer.add_summary(text, 0)
        # self.summary_writer.flush()
        return tl

    def plot_eer(self, acc):
        tl = [
            ['', '0', '30', '60', '90'],
            ['0', str(acc[0, 0]), str(acc[1, 0]), str(acc[2, 0]), str(acc[3, 0])],
            ['30', str(acc[0, 1]), str(acc[1, 1]), str(acc[2, 1]), str(acc[3, 1])],
            ['60', str(acc[0, 2]), str(acc[1, 2]), str(acc[2, 2]), str(acc[3, 2])],
            ['90', str(acc[0, 3]), str(acc[1, 3]), str(acc[2, 3]), str(acc[3, 3])]
        ]
        return tl

    def form_acc_to_feed(self):
        acc_to_feed = []
        for i in range(4):
            acc_to_feed.append(self.acc[i, i])
        return np.asarray(acc_to_feed)

    def form_eer_to_feed(self):
        eer_to_feed = []
        for i in range(4):
            eer_to_feed.append(self.eer[i, i])
        return np.asarray(eer_to_feed)

    def verify(self, data_reader, sess):
        angles = [0, 30, 60, 90]
        eer = np.zeros((4, 4))
        with tqdm(total=16) as pbar:
            for i in range(4):
                for j in range(4):
                    eer[i, j] = self.verify_probe_n_at_gallery_m(data_reader, sess, angles[i], angles[j])
                    pbar.update(1)
        print_acc(eer)
        result_txt = self.plot_eer(eer)
        return eer, result_txt

    def verify_probe_n_at_gallery_m(self, data_reader, sess, n, m):
        list_probe_indices = data_reader.get_list_of_probe_at_angle_k(angle_k=n)
        len_of_list = list_probe_indices.shape[0]
        embeddings_probe = np.zeros((len_of_list, self.embeddings_dims))
        idx_probe = []
        nof_batch_probe = int(np.floor(len_of_list / self.batch_size))
        for i in range(nof_batch_probe):
            x, _idx_probe, _angle_probe = data_reader.get_probe_by_indices(
                list_probe_indices[i * self.batch_size:(i + 1) * self.batch_size])
            x = np.expand_dims(np.array(x), -1)
            emb = sess.run(self.embeddings, feed_dict={self.input: x})
            embeddings_probe[i * self.batch_size:(i + 1) * self.batch_size, :] = emb
            idx_probe.append(_idx_probe)

        list_gallery_indices = data_reader.get_list_of_gallery_at_angle_k(angle_k=m)
        len_of_list = list_gallery_indices.shape[0]
        embeddings_gallery = np.zeros((len_of_list, self.embeddings_dims))
        idx_gallery = []
        nof_batch_gallery = int(np.floor(len_of_list / self.batch_size))
        for i in range(nof_batch_gallery):
            x, _idx_gallery, _angle_gallery = data_reader.get_gallery_by_indices(
                list_gallery_indices[i * self.batch_size:(i + 1) * self.batch_size])
            x = np.expand_dims(np.array(x), -1)
            emb = sess.run(self.embeddings, feed_dict={self.input: x})
            embeddings_gallery[i * self.batch_size:(i + 1) * self.batch_size, :] = emb
            idx_gallery.append(_idx_gallery)
        embeddings_probe = embeddings_probe[0:nof_batch_probe * self.batch_size, :]
        embeddings_gallery = embeddings_gallery[0:nof_batch_gallery * self.batch_size, :]
        predict = []
        gt = []
        idx_gallery = np.reshape(np.asarray(idx_gallery), (-1, 1))
        idx_probe = np.reshape(np.asarray(idx_probe), (-1, 1))
        for i in range(embeddings_probe.shape[0]):
            e = embeddings_probe[i, :]
            dis = np.sum(np.square(embeddings_gallery - e), 1)
            # label = np.zeros(embeddings_gallery.shape[0])
            # label[np.where(dis < 0.25)[0]] = 1
            # predict.append(label)
            predict.append(dis)
            _gt = np.ones(embeddings_gallery.shape[0])
            _gt[np.where(idx_gallery == idx_probe[i])[0]] = 0
            gt.append(_gt)
        predict = np.reshape(np.asarray(predict), (-1, 1))
        gt = np.reshape(np.asarray(gt), (-1, 1))
        fpr, tpr, thresholds = roc_curve(gt, predict, pos_label=1)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        # thresh = interp1d(fpr, thresholds)(eer)
        return eer


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    model = GaitGAN(tf_config=config)
    model.train(3)
