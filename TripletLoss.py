import os
import sys

import scipy.io as sio
import tensorflow as tf
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from sklearn.neighbors import NearestNeighbors
from tensorflow.contrib.tensorboard.plugins import projector
from tqdm import tqdm

from utils import DirGen
from utils import layers
from utils.data_reader import *
from utils.visualization_helper import *


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
        self.init_learning_rate = 5e-4
        self.beta1 = 0.5
        self.max_step = 100000
        self.batch_size = 100
        self.num_views = 14
        self.nof_person = 20
        self.nof_images = 6
        self.embeddings_dims = 52
        # build graph
        self.input = tf.placeholder(dtype=tf.float32,
                                    shape=(None, self.GEI_height, self.GEI_width, self.GEI_channel),
                                    name='input')
        self.sprites_input = tf.placeholder(dtype=tf.float32,
                                            shape=(None, self.GEI_height, self.GEI_width, self.GEI_channel),
                                            name='sprites_input')
        self.len_triplet = tf.placeholder(dtype=tf.int32, shape=())
        self.distance_pos = tf.placeholder(dtype=tf.float32, shape=(None), name='distance_pos_input')
        self.distance_neg = tf.placeholder(dtype=tf.float32, shape=(None), name='distance_neg_input')
        # ops of the network
        self.z = layers.encoder_old_school(self.input, self.embeddings_dims)
        self.z_sprite = layers.encoder_old_school(self.sprites_input, self.embeddings_dims, reuse=tf.AUTO_REUSE)
        # self.z = layers.encoder_deep_cnn(self.input, self.embeddings_dims)
        self.embeddings = tf.nn.l2_normalize(self.z, dim=1, epsilon=1e-12, name='embeddings')
        self.embeddings_sprite = tf.nn.l2_normalize(self.z_sprite, dim=1, epsilon=1e-12, name='embeddings_sprite')
        self.loss = layers.get_triplet_loss(self.embeddings, self.len_triplet)
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.compat.v1.train.exponential_decay(self.init_learning_rate,
                                                                  self.global_step,
                                                                  10000, 0.5, staircase=True)

        self.train_op = tf.train.MomentumOptimizer(momentum=0.9, learning_rate=self.learning_rate).minimize(self.loss,
                                                                                                            global_step=self.global_step)
        # tf.summary.histogram('embeddings', self.embeddings, collections=['general'])
        tf.summary.scalar('loss', self.loss, collections=['general'])
        tf.summary.scalar('nof_triplets', self.len_triplet, collections=['general'])
        with tf.name_scope('distances'):
            tf.summary.histogram('positive_batch', self.distance_pos, collections=['general'])
            tf.summary.histogram('negative_batch', self.distance_neg, collections=['general'])
        self.sum_op = tf.summary.merge_all(key='general')

        # self.paired_input = tf.placeholder(dtype=tf.float32,
        #                                    shape=(self.batch_size, self.GEI_height, self.GEI_width, self.GEI_channel),
        #                                    name='paired_input')
        # # self.input_for_dangle(dtype=tf.float32,
        # #                       shape=(self.batch_size, self.GEI_height, self.GEI_width, self.GEI_channel),
        # #                       name='input_for_dangle')
        # self.control_angle = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.num_views),
        #                                     name='control_angle')
        self.result_input = tf.placeholder(dtype=tf.string, name='results')
        self.result_input_eer = tf.placeholder(dtype=tf.string, name='results_eer')
        self.acc_input = tf.placeholder(dtype=tf.float32, shape=4, name='acc')
        self.eer_input = tf.placeholder(dtype=tf.float32, shape=4, name='eer')
        #
        # self.z = layers.encoder(self.input)
        # self.embeddings = tf.nn.l2_normalize(self.z, axis=1, epsilon=1e-12, name='embeddings')
        # self.generated_img = layers.generator(inputs=self.z, labels=self.control_angle)
        #
        # # logits for discriminator on id
        # _, self.logits_true_for_d = layers.pairwise_discriminator(self.input, self.input,
        #                                                           scope_name='discriminator_id',
        #                                                           reuse=tf.AUTO_REUSE)
        # _, self.logits_false_for_d = layers.pairwise_discriminator(self.input, self.generated_img,
        #                                                            scope_name='discriminator_id',
        #                                                            reuse=tf.AUTO_REUSE)
        # _, self.logits_true_for_g = layers.pairwise_discriminator(self.input, self.generated_img,
        #                                                           scope_name='discriminator_id',
        #                                                           reuse=tf.AUTO_REUSE)
        #
        # # logits for discriminator on angle
        # _, self.logits_true_for_d_angle = layers.conditional_discriminator(self.input, self.control_angle,
        #                                                                    scope_name='discriminator_angle')
        # _, self.logits_false_for_d_angle = layers.conditional_discriminator(self.generated_img, self.control_angle,
        #                                                                     scope_name='discriminator_angle',
        #                                                                     reuse=tf.AUTO_REUSE)
        # _, self.logits_true_for_g_angle = layers.conditional_discriminator(self.generated_img, self.control_angle,
        #                                                                    scope_name='discriminator_angle',
        #                                                                    reuse=tf.AUTO_REUSE)
        #
        # # get trainable variables
        # self.E_variable = [var for var in tf.trainable_variables() if var.name.startswith("encoder")]
        # self.G_variable = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
        # self.D_id_variable = [var for var in tf.trainable_variables() if var.name.startswith("discriminator_id")]
        # self.D_angle_variable = [var for var in tf.trainable_variables() if var.name.startswith("discriminator_angle")]
        #
        # # define losses
        # self.D_id_loss_true = -tf.reduce_mean(tf.log(self.logits_true_for_d))
        # self.D_id_loss_false = -tf.reduce_mean(tf.log(1.0 - self.logits_false_for_d))
        # self.D_id_loss = self.D_id_loss_true + self.D_id_loss_false
        #
        # self.D_angle_loss_true = -tf.reduce_mean(tf.log(self.logits_true_for_d_angle))
        # self.D_angle_loss_false = -tf.reduce_mean(tf.log(1.0 - self.logits_false_for_d_angle))
        # self.D_angle_loss = self.D_angle_loss_true + self.D_angle_loss_false
        #
        # self.G_loss_id = -tf.reduce_mean(tf.log(self.logits_true_for_g))
        # self.G_loss_angle = -tf.reduce_mean(tf.log(self.logits_true_for_g_angle))
        # self.G_loss = self.G_loss_id + self.G_loss_angle
        # self.Recon_loss = tf.reduce_mean(tf.losses.absolute_difference(
        #     self.input, self.generated_img)) * 2
        #
        # self.D_id_optimizer = tf.train.AdamOptimizer(beta1=self.beta1, learning_rate=self.learning_rate * 0.05)
        # self.D_id_grads_and_vars = self.D_id_optimizer.compute_gradients(self.D_id_loss, var_list=self.D_id_variable)
        # self.D_id_train = self.D_id_optimizer.apply_gradients(self.D_id_grads_and_vars)
        #
        # self.D_angle_optimizer = tf.train.AdamOptimizer(beta1=self.beta1, learning_rate=self.learning_rate * 0.05)
        # self.D_angle_grads_and_vars = self.D_angle_optimizer.compute_gradients(self.D_angle_loss,
        #                                                                        var_list=self.D_angle_variable)
        # self.D_angle_train = self.D_angle_optimizer.apply_gradients(self.D_angle_grads_and_vars)
        #
        # self.EG_optimizer = tf.train.AdamOptimizer(self.learning_rate, self.beta1)
        # self.EG_grads_and_vars = self.EG_optimizer.compute_gradients(self.G_loss + self.Recon_loss,
        #                                                              var_list=[self.E_variable, self.G_variable])
        # self.EG_train = self.EG_optimizer.apply_gradients(self.EG_grads_and_vars)
        #
        # # Summary Registration
        # self.d_id_loss_sum = tf.summary.scalar('D_id_loss', self.D_id_loss, collections=['general'])
        # self.d_id_loss_true_sum = tf.summary.scalar('D_id_loss_true', self.D_id_loss_true, collections=['general'])
        # self.d_id_loss_false_sum = tf.summary.scalar('D_id_loss_false', self.D_id_loss_false, collections=['general'])
        #
        # self.d_angle_loss_sum = tf.summary.scalar('D_angle_loss', self.D_angle_loss, collections=['general'])
        # self.d_angle_loss_true_sum = tf.summary.scalar('D_angle_loss_true', self.D_angle_loss_true,
        #                                                collections=['general'])
        # self.d_angle_loss_false_sum = tf.summary.scalar('D_angle_loss_false', self.D_angle_loss_false,
        #                                                 collections=['general'])
        #
        # self.sum_eg_loss = tf.summary.scalar('EG_loss', self.G_loss + self.Recon_loss, collections=['general'])
        # self.sum_recon_loss = tf.summary.scalar('Recon_loss', self.Recon_loss, collections=['general'])
        # self.sum_g_loss = tf.summary.scalar('G_loss', self.G_loss, collections=['general'])
        # self.sum_g_loss_id = tf.summary.scalar('G_loss_id', self.G_loss_id, collections=['general'])
        # self.sum_g_loss_angle = tf.summary.scalar('G_loss_angle', self.G_loss_angle, collections=['general'])
        #
        # self.sum_input = tf.summary.image('input', self.input, collections=['general'])
        # # self.sum_paired_input = tf.summary.image('paried_input', self.paired_input, collections=['general'])
        # self.sum_generated = tf.summary.image('generated', self.generated_img, collections=['general'])
        #

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

        # self.embedding visualization
        dr = DataReader(root_path='./GEI/GEI', usage='Training')
        x_to_sprites, labels, _, k = dr.select_id_v2(8, 8)
        while k < 8:
            x_to_sprites, labels, _, k = dr.select_id_v2(8, 8)
        del dr
        self.x_to_sprites = np.asarray(x_to_sprites)

        script_name = sys.argv[0][:-3].split('/')[-1]
        self.logdir = DirGen.dir_gen('./exp/summary/' + script_name)
        # self.modeldir = DirGen.dir_gen('./exp/model/')
        self.modeldir = self.logdir
        with open(os.path.join(self.logdir, 'log.txt'), 'w') as file:
            file.write(script_name + '\n')
            file.write("learning rate:" + str(self.learning_rate) + '\n')
            file.close()

        os.system('cp ' + script_name + '.py ' + self.logdir)
        path_for_sprites = './sprite_images.png'
        path_for_meta = './metadata.tsv'
        write_sprite_image(path_for_sprites, self.x_to_sprites)
        write_metadata(path_for_meta, labels)
        tensor_shape = (64, self.embeddings_dims)
        self.embedding_var = tf.Variable(tf.zeros(tensor_shape), name='z_feature')
        self.embedding_assign = self.embedding_var.assign(self.embeddings_sprite)
        self.pcf = projector.ProjectorConfig()
        self.embedding = self.pcf.embeddings.add()
        self.embedding.tensor_name = self.embedding_var.name
        self.embedding.metadata_path = path_for_meta
        self.embedding.sprite.image_path = path_for_sprites
        self.embedding.sprite.single_image_dim.extend([88, 128])

    def test(self):
        with tf.Session(config=self.tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            # self.pcf.model_checkpoint_path = os.path.join(self.modeldir, 'model-ckpt')
            self.summary_writer = tf.summary.FileWriter(graph=sess.graph, logdir=self.logdir)
            self.saver = tf.train.Saver(max_to_keep=200)
            self.saver.restore(sess, 'C:/Users/bingz/Dropbox/GoodModel/model/20190903-013700/-39999')
            dr = DataReader('./GEI/GEI')
            angles = [0, 30, 60, 90]
            with tqdm(total=16) as pbar:
                for i in range(4):
                    for j in range(4):
                        self.extract_feature_probe_n_at_gallery_m(dr, sess, angles[i], angles[j])
                        pbar.update(1)

    def train(self):
        with tf.Session(config=self.tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            # self.pcf.model_checkpoint_path = os.path.join(self.modeldir, 'model-ckpt')
            self.summary_writer = tf.summary.FileWriter(graph=sess.graph, logdir=self.logdir)
            self.saver = tf.train.Saver(max_to_keep=200)
            projector.visualize_embeddings(self.summary_writer, self.pcf)
            # self.saver.restore(sess, './exp/model/20190903-011955/-999')
            self.result_txt = 'Not updated yet'
            self.result_txt_eer = 'Not updated yet'
            self.acc = np.zeros((4, 4))
            self.eer = np.zeros((4, 4))

            dr = DataReader('./GEI/GEI')
            for step in tqdm(range(self.max_step)):
                x, label, paths, k = dr.select_id_v2(self.nof_person, self.nof_images)
                while k < self.nof_person:
                    x, label, paths, k = dr.select_id_v2(self.nof_person, self.nof_images)
                x = np.expand_dims(np.array(x), -1)
                embeddings = sess.run(self.embeddings, feed_dict={self.input: x})
                triplet, dist_p, dist_n = self.select_triplets(embeddings, k, self.nof_images, 0.2)
                # label_t = np.array(label)[triplet][:]
                paths_t = []
                for t in triplet:
                    a = t[0]
                    p = t[1]
                    n = t[2]
                    paths_t.append(paths[a])
                    paths_t.append(paths[p])
                    paths_t.append(paths[n])

                nof_triplet = len(triplet) * 3
                if nof_triplet > 0:
                    x = dr.read_from_paths(paths_t)
                    x = np.expand_dims(np.array(x), -1)
                    _, loss, sum, _ = sess.run([self.train_op, self.loss, self.sum_op, self.embedding_assign],
                                               feed_dict={self.input: x,
                                                          self.sprites_input: np.expand_dims(self.x_to_sprites, -1),
                                                          self.len_triplet: nof_triplet,
                                                          self.distance_neg: dist_n, self.distance_pos: dist_p})
                    print('step:%d, loss:%lf, nof_triplet:%d' % (step, loss, nof_triplet))
                    self.summary_writer.add_summary(sum, step)

                if (step) % 2000 == 0:
                    if not os.path.exists(self.modeldir):
                        os.makedirs(self.modeldir)
                    self.saver.save(sess=sess, save_path=os.path.join(self.modeldir, 'model-ckpt'), global_step=step)

                # if (step) % 2000 == 0:
                #     print('validating ...')
                #     self.acc, self.result_txt = self.validate(dr, sess)
                #     self.eer, self.result_txt_eer = self.verify(dr, sess)
                #     sum = sess.run(self.summary_op_validate, feed_dict={self.acc_input: self.form_acc_to_feed(),
                #                                                         self.result_input: self.result_txt,
                #                                                         self.eer_input: self.form_eer_to_feed(),
                #                                                         self.result_input_eer: self.result_txt_eer})
                #     self.summary_writer.add_summary(sum, step)
                # acc_path = 'acc_%d.mat' % step
                # sio.savemat(join(logdir, acc_path), {'acc': self.acc})

            del dr

    def select_triplets(self, embeddings, nof_person, nof_images, delta):
        # aff = []
        triplet = []
        pos_dist_to_check = []
        neg_dist_to_check = []
        for anchor_id in range(nof_person * nof_images):
            dist = np.sum(np.square(embeddings - embeddings[anchor_id]), 1)
            neg_dist = np.copy(dist)
            neg_dist[anchor_id:(anchor_id // nof_images + 1) * nof_images] = np.NAN
            # aff.append(dist)
            for pos_id in range(anchor_id + 1, (anchor_id // nof_images + 1) * nof_images):
                neg_ids = np.where(neg_dist - dist[pos_id] < delta)[0]
                nof_neg_ids = len(neg_ids)
                pos_dist_to_check.append(dist[pos_id])
                if nof_neg_ids > 10:
                    rand_id = np.random.randint(nof_neg_ids)
                    neg_id = neg_ids[rand_id]
                    triplet.append([anchor_id, pos_id, neg_id])

            neg_dist_to_check.append(neg_dist)
        temp = []
        for i, piece in enumerate(neg_dist_to_check):
            idx = np.ones(60)
            idx[0:(i // nof_images + 1) * nof_images] = 0
            temp = np.concatenate([temp, piece[np.where(idx == 1)[0]]])
        neg_dist_to_check = temp
        pos_dist_to_check = np.asarray(pos_dist_to_check)
        np.random.shuffle(triplet)
        return triplet, pos_dist_to_check, neg_dist_to_check

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
        # probe_feature_path = 'probe%d.mat' % n
        # gallery_feature_path = 'gallery%d.mat' % m
        # sio.savemat(probe_feature_path, {'probe_feature': embeddings_probe, 'probe_id': idx_probe})
        # sio.savemat(gallery_feature_path, {'gallery_feature': embeddings_gallery, 'probe_id': idx_gallery})
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(embeddings_gallery[0:nof_batch_gallery * self.batch_size, :])
        distances, indices = nn.kneighbors(embeddings_probe[0:nof_batch_probe * self.batch_size, :])

        idx_gallery = np.reshape(np.asarray(idx_gallery), (-1, 1))
        idx_probe = np.reshape(np.asarray(idx_probe), (-1, 1))
        predicted = idx_gallery[indices, :]

        diff = np.reshape(predicted, (-1, 1)) - idx_probe[0:nof_batch_probe * self.batch_size, :]
        acc = np.where(diff == 0)[0].shape[0] / float(nof_batch_probe * self.batch_size)
        return acc

    def extract_feature_probe_n_at_gallery_m(self, data_reader, sess, n, m):
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
        probe_feature_path = 'probe%d.mat' % n
        gallery_feature_path = 'gallery%d.mat' % m
        sio.savemat(probe_feature_path, {'probe_feature': embeddings_probe, 'probe_id': idx_probe})
        sio.savemat(gallery_feature_path, {'gallery_feature': embeddings_gallery, 'probe_id': idx_gallery})
        return

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
            # label[np.where(dis < 0.5)[0]] = 1
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
    # model.train()
    model.test()