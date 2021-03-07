import os

import numpy as np
import tensorflow as tf

from utils import layers
from utils.data_reader import *
from utils import DirGen
from tqdm import tqdm
# from PIL import Image
from scipy.misc import imsave

class GaitGAN():
    def __init__(self):
        self.GEI_height = 128
        self.GEI_width = 88
        self.GEI_channel = 1
        self.learning_rate = 1e-5
        self.beta1 = 0.5
        self.max_step = 30000
        self.batch_size = 100
        self.num_views = 28
        dr = DataReader('../GaitGAN/GEI')
        del dr
        # build graph
        self.probe = tf.placeholder(dtype=tf.float32,
                                    shape=(self.batch_size, self.GEI_height, self.GEI_width, self.GEI_channel),
                                    name='probe')
        self.probe_generated = tf.placeholder(dtype=tf.float32,
                                              shape=(
                                              self.batch_size, self.GEI_height, self.GEI_width, self.GEI_channel),
                                              name='probe_generated')

        self.control_angle = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.num_views),
                                            name='control_angle')

        self.input_gei = tf.placeholder(dtype=tf.float32,
                                        shape=(self.batch_size, self.GEI_height, self.GEI_width, self.GEI_channel))
        self.input_label = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.num_views))
        self.input_ground_truth = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, 1))

        self.z = layers.encoder(self.probe)
        self.embeddings = tf.nn.l2_normalize(self.z, axis=1, epsilon=1e-12, name='embeddings')
        self.generated_img = layers.generator(inputs=self.z, labels=self.control_angle)

        # define logits
        _, self.logits_true_for_d = layers.pairwise_discriminator(self.probe, self.probe_generated,
                                                                  scope_name='discriminator_id')
        _, self.logits_false_for_d = layers.pairwise_discriminator(self.probe, self.generated_img,
                                                                   scope_name='discriminator_id',
                                                                   reuse=tf.AUTO_REUSE)
        _, self.logits_true_for_g = layers.pairwise_discriminator(self.probe, self.generated_img,
                                                                  scope_name='discriminator_id',
                                                                  reuse=tf.AUTO_REUSE)

        _, self.logits_true_for_d_angle = layers.conditional_discriminator(self.probe_generated, self.control_angle,
                                                                           scope_name='discriminator_angle')
        _, self.logits_false_for_d_angle = layers.conditional_discriminator(self.generated_img, self.control_angle,
                                                                            scope_name='discriminator_angle',
                                                                            reuse=tf.AUTO_REUSE)
        _, self.logits_true_for_g_angle = layers.conditional_discriminator(self.generated_img, self.control_angle,
                                                                           scope_name='discriminator_angle',
                                                                           reuse=tf.AUTO_REUSE)
        self.logits, _ = layers.conditional_discriminator(self.input_gei, self.input_label,
                                                          scope_name='discriminator_angle',
                                                          reuse=tf.AUTO_REUSE)

        self.E_variable = [var for var in tf.trainable_variables() if var.name.startswith("encoder")]
        self.G_variable = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
        self.D_id_variable = [var for var in tf.trainable_variables() if var.name.startswith("discriminator_id")]
        self.D_angle_variable = [var for var in tf.trainable_variables() if var.name.startswith("discriminator_angle")]

        self.D_id_loss = -tf.reduce_mean(tf.log(self.logits_true_for_d) + tf.log(1.0 - self.logits_false_for_d))
        self.D_angle_loss = -tf.reduce_mean(
            tf.log(self.logits_true_for_d_angle) + tf.log(
                1.0 - self.logits_false_for_d_angle)) + tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                                                    labels=self.input_ground_truth))
        self.G_loss = -tf.reduce_mean(tf.log(self.logits_true_for_g)) - tf.reduce_mean(
            tf.log(self.logits_true_for_g_angle))
        self.Recon_loss = tf.reduce_mean(tf.losses.absolute_difference(
            self.probe_generated, self.generated_img)) * 2

        self.D_id_optimizer = tf.train.AdamOptimizer(beta1=self.beta1, learning_rate=self.learning_rate * 0.05)
        self.D_id_grads_and_vars = self.D_id_optimizer.compute_gradients(self.D_id_loss, var_list=self.D_id_variable)
        self.D_id_train = self.D_id_optimizer.apply_gradients(self.D_id_grads_and_vars)

        self.D_angle_optimizer = tf.train.AdamOptimizer(beta1=self.beta1, learning_rate=self.learning_rate * 0.05)
        self.D_angle_grads_and_vars = self.D_angle_optimizer.compute_gradients(self.D_angle_loss,
                                                                               var_list=self.D_angle_variable)
        self.D_angle_train = self.D_angle_optimizer.apply_gradients(self.D_angle_grads_and_vars)

        self.EG_optimizer = tf.train.AdamOptimizer(self.learning_rate, self.beta1)
        self.EG_grads_and_vars = self.EG_optimizer.compute_gradients(self.G_loss + self.Recon_loss,
                                                                     var_list=[self.E_variable, self.G_variable])
        self.EG_train = self.EG_optimizer.apply_gradients(self.EG_grads_and_vars)

        self.d_id_loss_sum = tf.summary.scalar('D_id_loss', self.D_id_loss)
        self.d_angle_loss_sum = tf.summary.scalar('D_angle_loss', self.D_angle_loss)
        self.sum_g_loss = tf.summary.scalar('EG_loss', self.G_loss)
        self.sum_recon_loss = tf.summary.scalar('Recon_loss', self.Recon_loss)
        self.sum_degree_0 = tf.summary.image('degree_0', self.probe)
        self.sum_degree_45 = tf.summary.image('ground_truth', self.probe_generated)
        self.sum_faked = tf.summary.image('faked', self.generated_img)
        self.summary_op = tf.summary.merge_all()

    def train(self, k):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            logdir = DirGen.dir_gen('./final/summary/MO/')
            modeldir = DirGen.dir_gen('./final/model/MO/')
            self.summary_writer = tf.summary.FileWriter(graph=sess.graph, logdir=logdir)
            self.saver = tf.train.Saver(max_to_keep=200)
            import os
            print(os.getcwd())
            data_reader = DataReader14Views.DataReader(root_path='../../Workspace/GaitGAN/')
            # probe_data_reader = DataReaderTripletCasia_b.DataReader(root_path='./GEI_CASIA_B/', data_info='probe.mat')
            # gallery_data_reader = DataReaderTripletCasia_b.DataReader(root_path='./GEI_CASIA_B/',
            #                                                           data_info='gallery.mat')
            for step in tqdm(range(self.max_step)):
                x, y, control_angle = data_reader.nextBatchforControl(self.batch_size)
                x = np.expand_dims(np.array(x), -1)
                y = np.expand_dims(np.array(y), -1)
                control_angle = np.squeeze(control_angle, 1)
                for j in range(k):
                    _, loss_D, sum = sess.run(
                        [self.D_id_train, self.D_id_loss, self.d_id_loss_sum],
                        feed_dict={self.probe: x, self.probe_generated: y, self.control_angle: control_angle})

                gei, label = data_reader.nextBatchforDAngle(50)
                gei = np.expand_dims(np.array(gei), -1)
                label = np.squeeze(np.array(label), 1)
                gt = np.zeros((self.batch_size, 1))
                for i in range(self.batch_size):
                    if i % 2 == 0:
                        gt[i] = 1
                    else:
                        gt[i] = 0
                # for j in range(k):
                #     _, loss_D_angle, sum = sess.run([self.D_angle_train, self.D_angle_loss, self.d_angle_loss_sum],
                #                                     feed_dict={self.input_gei: gei, self.input_label: label,
                #                                                self.input_ground_truth: gt, self.probe: x,
                #                                                self.probe_generated: y,
                #                                                self.control_angle: control_angle})
                _, loss_EG, sum = sess.run([self.EG_train, self.G_loss, self.summary_op],
                                           feed_dict={self.input_gei: gei, self.input_label: label,
                                                      self.input_ground_truth: gt, self.probe: x,
                                                      self.probe_generated: y,
                                                      self.control_angle: control_angle})
                # print("Step: %d\t Loss_G: %lf\t Loss_D: %lf\t Loss_Angle: %lf" % (step, loss_EG, loss_D, loss_D_angle))
                print("Step: %d\t Loss_G: %lf\t Loss_D: %lf" % (step, loss_EG, loss_D))
                self.summary_writer.add_summary(sum, step)

                if (step + 1) % 1000 == 0:
                    if not os.path.exists(modeldir):
                        os.makedirs(modeldir)
                    self.saver.save(sess=sess, save_path=modeldir + '/', global_step=step)
                if (step - 1) % 400 == 0:
                    if not os.path.exists('./final/output/MO/save'):
                        os.makedirs('./final/output/MO/save')
                    self.test(step, sess)

                # if (step - 1) % 400 == 0:
                #     probe_data_reader.index = 0
                #     gallery_data_reader.index = 0
                #
                #     probe_feature = []
                #     l = int(np.floor(probe_data_reader.total_count / 100) +1)
                #     for _ in range(l):
                #         x = probe_data_reader.getBatchforTest(100)
                #         x = np.expand_dims(np.array(x), -1)
                #         feature_ = sess.run(self.embeddings, feed_dict={self.x: x})
                #         probe_feature.append(feature_)
                #     probe_feature = np.concatenate(probe_feature)
                #
                #     gallery_feature = []
                #     l = int(np.floor(gallery_data_reader.total_count / 100)+1)
                #     for _ in range(l):
                #         x = gallery_data_reader.getBatchforTest(100)
                #         x = np.expand_dims(np.array(x), -1)
                #         feature_ = sess.run(self.embeddings, feed_dict={self.x: x})
                #         gallery_feature.append(feature_)
                #     gallery_feature = np.concatenate(gallery_feature)
                #
                #     probe_angle = ['054', '090', '126']
                #     gallery_angle = ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180']
                #     gallery = sio.loadmat('./utils/gallery.mat')
                #     probe = sio.loadmat('./utils/probe.mat')
                #     acc = []
                #     acc_mean = []
                #     for l in range(3):
                #         probe_ids = [i for i in range(probe['path'].size) if probe['path'][i][26:29] == probe_angle[l]]
                #         gallery_ids = [i for i in range(gallery['path'].size) if
                #                        not gallery['path'][i][26:29] == probe_angle[l]]
                #         pf = probe_feature[probe_ids, :]
                #         gf = gallery_feature[gallery_ids, :]
                #         gf_ids = gallery['ids'][gallery_ids]
                #         pf_ids = probe['ids'][probe_ids]
                #         nbrs = NearestNeighbors(n_neighbors=1, p=2).fit(gf)
                #         distances, indices = nbrs.kneighbors(pf)
                #         prediction = gf_ids[indices]
                #         ground_truth = pf_ids
                #         diff = np.squeeze(prediction, axis=-1) - ground_truth
                #         acc.append(np.where(diff == 0)[0].size / float(ground_truth.size))
                #     for l in range(3):
                #         acc_l=[]
                #         for m in range(11):
                #             if probe_angle[l] == gallery_angle[m]:
                #                 continue
                #             else:
                #                 probe_ids = [i for i in range(probe['path'].size) if
                #                              probe['path'][i][26:29] == probe_angle[l]]
                #                 gallery_ids = [i for i in range(gallery['path'].size) if
                #                                gallery['path'][i][26:29] == gallery_angle[m]]
                #                 pf = probe_feature[probe_ids, :]
                #                 gf = gallery_feature[gallery_ids, :]
                #                 gf_ids = gallery['ids'][gallery_ids]
                #                 pf_ids = probe['ids'][probe_ids]
                #                 nbrs = NearestNeighbors(n_neighbors=1, p=2).fit(gf)
                #                 distances, indices = nbrs.kneighbors(pf)
                #                 prediction = gf_ids[indices]
                #                 ground_truth = pf_ids
                #                 diff = np.squeeze(prediction, axis=-1) - ground_truth
                #                 acc_l.append(np.where(diff==0)[0].size  / float(ground_truth.size))
                #         acc_mean.append(np.mean(acc_l))
                #
                #     summary_1 = tf.Summary(value=[tf.Summary.Value(tag='054', simple_value=acc[0])])
                #     summary_2 = tf.Summary(value=[tf.Summary.Value(tag='090', simple_value=acc[1])])
                #     summary_3 = tf.Summary(value=[tf.Summary.Value(tag='126', simple_value=acc[2])])
                #     summary_4 = tf.Summary(value=[tf.Summary.Value(tag='A054', simple_value=acc_mean[0])])
                #     summary_5 = tf.Summary(value=[tf.Summary.Value(tag='A090', simple_value=acc_mean[1])])
                #     summary_6 = tf.Summary(value=[tf.Summary.Value(tag='A126', simple_value=acc_mean[2])])
                #     self.summary_writer.add_summary(summary_1, step)
                #     self.summary_writer.add_summary(summary_2, step)
                #     self.summary_writer.add_summary(summary_3, step)
                #     self.summary_writer.add_summary(summary_4, step)
                #     self.summary_writer.add_summary(summary_5, step)
                #     self.summary_writer.add_summary(summary_6, step)
                #     self.summary_writer.flush()

    def test(self, step, sess):
        data_reader = DataReader14Views.DataReader(root_path='../../Workspace/GaitGAN/')
        x = data_reader.getTestSample()
        x = np.expand_dims(x, axis=-1)
        x = np.tile(x, (1, self.batch_size, 1, 1))
        x = np.reshape(x, (self.batch_size, 128, 88, 1))
        # for i in range(self.num_views):
        #     control_angle = np.zeros((self.batch_size, self.num_views))
        #     control_angle[:, i] = 1
        #     img = sess.run(self.generated_img, feed_dict={self.probe: x, self.control_angle: control_angle})
        #     img = img[0, :, :, :]
        #     img = np.squeeze(img, axis=-1)
        #     filename = './final/output/MO/save/' + str(step) + '_' + str(i) + '.bmp'
        #     im = Image.fromarray(img)
        #     im = im.convert(mode='L')
        #     im.save(filename)
        for i in range(self.num_views):
            control_angle = np.zeros((self.batch_size, self.num_views))
            control_angle[:, i] = 1
            img = sess.run(self.generated_img, feed_dict={self.probe: x, self.control_angle: control_angle})
            img = img[0, :, :, :]
            img = np.squeeze(img, axis=-1)
            filename = './final/output/MO/save/' + str(step) + '_' + str(i) + '.png'
            imsave(filename, img)

    # def generate(self):
    #     with tf.Session() as sess:
    #         sess.run(tf.global_variables_initializer())
    #         saver = tf.train.Saver()
    #         saver.restore(sess, './AAAI/14Views/model/20180904-200402/-29999')
    #
    #         data_reader = DataReader14Views.DataReader(root_path='../GaitGAN/')
    #         # for i in range(4):
    #         x = data_reader.getTestSample(k=12, id=7766)
    #         x = np.expand_dims(x, axis=-1)
    #         x = np.tile(x, (1, self.batch_size, 1, 1))
    #         x = np.reshape(x, (self.batch_size, 128, 88, 1))
    #         # for j in range(4):
    #         control_angle = np.zeros((self.batch_size, self.num_views))
    #         control_angle[:, 13] = 1
    #         img = sess.run(self.generated_img, feed_dict={self.x: x, self.control_angle: control_angle})
    #         img = img[0, :, :, :]
    #         # img = np.squeeze(img, axis=0)
    #         img = np.squeeze(img, axis=-1)
    #         filename = './exp1/' + str(6) + '_' + str(7766) + '.png'
    #         imsave(filename, img)


if __name__ == '__main__':
    model = GaitGAN()
    model.train(3)
    # model.generate()
    # model.test_all()
    # model.extract_features()
    # model.test_generate()
