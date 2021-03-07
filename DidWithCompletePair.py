import tensorflow as tf

from utils import layers
from utils.data_reader import *
from utils import DirGen
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import scipy.io as sio
import sys
import os
import validator
from scipy.misc import imsave


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
        self.learning_rate = 1e-5
        self.beta1 = 0.5
        self.max_step = 40000
        self.batch_size = 100
        self.num_views = 14

        # build graph
        self.input = tf.placeholder(dtype=tf.float32,
                                    shape=(self.batch_size, self.GEI_height, self.GEI_width, self.GEI_channel),
                                    name='input')
        self.paired_input = tf.placeholder(dtype=tf.float32,
                                           shape=(self.batch_size, self.GEI_height, self.GEI_width, self.GEI_channel),
                                           name='paired_input')
        self.control_angle = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.num_views),
                                            name='control_angle')
        self.result_input = tf.placeholder(dtype=tf.string, name='results')
        self.acc_input = tf.placeholder(dtype=tf.float32, shape=4, name='acc')

        self.z = layers.encoder(self.input)
        self.embeddings = tf.nn.l2_normalize(self.z, axis=1, epsilon=1e-12, name='embeddings')
        self.generated_img = layers.generator(inputs=self.z, labels=self.control_angle)

        # logits for discriminator on id
        _, self.logits_true_for_d = layers.pairwise_discriminator(self.input, self.paired_input,
                                                                  scope_name='discriminator_id',
                                                                  reuse=tf.AUTO_REUSE)
        _, self.logits_false_for_d = layers.pairwise_discriminator(self.input, self.generated_img,
                                                                   scope_name='discriminator_id',
                                                                   reuse=tf.AUTO_REUSE)
        _, self.logits_true_for_g = layers.pairwise_discriminator(self.input, self.generated_img,
                                                                  scope_name='discriminator_id',
                                                                  reuse=tf.AUTO_REUSE)

        self.E_variable = [var for var in tf.trainable_variables() if var.name.startswith("encoder")]
        self.G_variable = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
        self.D_id_variable = [var for var in tf.trainable_variables() if var.name.startswith("discriminator_id")]

        self.D_id_loss_true = -tf.reduce_mean(tf.log(self.logits_true_for_d))
        self.D_id_loss_false = -tf.reduce_mean(tf.log(1.0 - self.logits_false_for_d))
        self.D_id_loss = self.D_id_loss_true + self.D_id_loss_false

        self.G_loss_id = -tf.reduce_mean(tf.log(self.logits_true_for_g))
        self.G_loss = self.G_loss_id
        self.Recon_loss = tf.reduce_mean(tf.losses.absolute_difference(
            self.paired_input, self.generated_img)) * 2

        self.D_id_optimizer = tf.train.AdamOptimizer(beta1=self.beta1, learning_rate=self.learning_rate * 0.05)
        self.D_id_grads_and_vars = self.D_id_optimizer.compute_gradients(self.D_id_loss, var_list=self.D_id_variable)
        self.D_id_train = self.D_id_optimizer.apply_gradients(self.D_id_grads_and_vars)

        self.EG_optimizer = tf.train.AdamOptimizer(self.learning_rate, self.beta1)
        self.EG_grads_and_vars = self.EG_optimizer.compute_gradients(self.G_loss + self.Recon_loss,
                                                                     var_list=[self.E_variable, self.G_variable])
        self.EG_train = self.EG_optimizer.apply_gradients(self.EG_grads_and_vars)

        # Summary Registration
        self.d_id_loss_sum = tf.summary.scalar('D_id_loss', self.D_id_loss, collections=['general'])
        self.d_id_loss_true_sum = tf.summary.scalar('D_id_loss_true', self.D_id_loss_true, collections=['general'])
        self.d_id_loss_false_sum = tf.summary.scalar('D_id_loss_false', self.D_id_loss_false, collections=['general'])

        self.sum_eg_loss = tf.summary.scalar('EG_loss', self.G_loss + self.Recon_loss, collections=['general'])
        self.sum_recon_loss = tf.summary.scalar('Recon_loss', self.Recon_loss, collections=['general'])
        self.sum_g_loss = tf.summary.scalar('G_loss', self.G_loss, collections=['general'])
        self.sum_g_loss_id = tf.summary.scalar('G_loss_id', self.G_loss_id, collections=['general'])

        self.sum_input = tf.summary.image('input', self.input, collections=['general'])
        self.sum_paired_input = tf.summary.image('paried_input', self.paired_input, collections=['general'])
        self.sum_generated = tf.summary.image('generated', self.generated_img, collections=['general'])

        self.sum_result = tf.summary.text('result', self.result_input, collections=['validate'])
        self.sum_acc0 = tf.summary.scalar('acc_0_0', self.acc_input[0], collections=['validate'])
        self.sum_acc0 = tf.summary.scalar('acc_30_30', self.acc_input[1], collections=['validate'])
        self.sum_acc0 = tf.summary.scalar('acc_60_60', self.acc_input[2], collections=['validate'])
        self.sum_acc0 = tf.summary.scalar('acc_90_90', self.acc_input[3], collections=['validate'])
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
                file.write("EG Optimizer:" + self.EG_optimizer.get_name() + '\n')
                file.write("Did Optimizer:" + self.D_id_optimizer.get_name() + '\n')
                file.write("beta1:" + str(self.beta1) + '\n')
                file.close()
            self.summary_writer = tf.summary.FileWriter(graph=sess.graph, logdir=logdir)
            self.saver = tf.train.Saver(max_to_keep=200)
            self.result_txt = 'Not updated yet'
            self.acc = np.zeros((4, 4))

            dr = DataReader(root_path='../GaitGAN/GEI', usage='Training')
            for step in tqdm(range(self.max_step)):
                angle_list = np.random.randint(0, 14, self.batch_size)
                x, paired_x = dr.get_next_batch_at_given_angle(self.batch_size, angle_list)
                x = np.expand_dims(np.array(x), -1)
                paired_x = np.expand_dims(np.array(paired_x), -1)
                angle = []
                for _a in angle_list:
                    temp = np.zeros(self.num_views)
                    temp[_a] = 1
                    angle.append(temp)
                angle = np.array(angle)
                for j in range(k):
                    _, loss_D = sess.run(
                        [self.D_id_train, self.D_id_loss],
                        feed_dict={self.input: x, self.paired_input: paired_x, self.control_angle: angle})

                _, loss_EG, sum = sess.run([self.EG_train, self.G_loss, self.summary_op_general],
                                           feed_dict={self.input: x, self.control_angle: angle,
                                                      self.paired_input: paired_x})
                print("Step: %d\t Loss_G: %lf\t  Loss_D: %lf" % (step, loss_EG, loss_D))
                self.summary_writer.add_summary(sum, step)

                if (step + 1) % 1000 == 0:
                    if not os.path.exists(modeldir):
                        os.makedirs(modeldir)
                    self.saver.save(sess=sess, save_path=modeldir + '/', global_step=step)

                if (step + 1) % 1000 == 0:
                    print('validating ...')
                    self.acc, self.result_txt = self.validate(dr, sess)
                    sum = sess.run(self.summary_op_validate,
                                   feed_dict={self.result_input: self.result_txt, self.acc_input: self.form_acc_to_feed()})
                    self.summary_writer.add_summary(sum, step)
                    acc_path = 'acc_%d.mat' % step
                    sio.savemat(join(logdir, acc_path), {'acc': self.acc})

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
        result_txt = self.plot_acc(acc, sess)
        return acc, result_txt

    def validate_probe_n_at_gallery_m(self, data_reader, sess, n, m):
        list_probe_indices = data_reader.get_list_of_probe_at_angle_k(angle_k=n)
        len_of_list = list_probe_indices.shape[0]
        embeddings_probe = np.zeros((len_of_list, 128))
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
        embeddings_gallery = np.zeros((len_of_list, 128))
        idx_gallery = []
        nof_batch_gallery = int(np.floor(len_of_list / self.batch_size))
        for i in range(nof_batch_gallery):
            x, _idx_gallery, _angle_gallery = data_reader.get_gallery_by_indices(
                list_gallery_indices[i * self.batch_size:(i + 1) * self.batch_size])
            x = np.expand_dims(np.array(x), -1)
            emb = sess.run(self.embeddings, feed_dict={self.input: x})
            embeddings_gallery[i * self.batch_size:(i + 1) * self.batch_size, :] = emb
            idx_gallery.append(_idx_gallery)
        #
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(embeddings_gallery[0:nof_batch_gallery * self.batch_size, :])
        distances, indices = nn.kneighbors(embeddings_probe[0:nof_batch_probe * self.batch_size, :])

        idx_gallery = np.reshape(np.asarray(idx_gallery), (-1, 1))
        idx_probe = np.reshape(np.asarray(idx_probe), (-1, 1))
        predicted = idx_gallery[indices, :]

        diff = np.reshape(predicted, (-1, 1)) - idx_probe[0:nof_batch_probe * self.batch_size, :]
        acc = np.where(diff == 0)[0].shape[0] / float(nof_batch_probe * self.batch_size)
        return acc

    #
    #         #     summary_1 = tf.Summary(value=[tf.Summary.Value(tag='054', simple_value=acc[0])])
    #         #     summary_2 = tf.Summary(value=[tf.Summary.Value(tag='090', simple_value=acc[1])])
    #         #     summary_3 = tf.Summary(value=[tf.Summary.Value(tag='126', simple_value=acc[2])])
    #         #     summary_4 = tf.Summary(value=[tf.Summary.Value(tag='A054', simple_value=acc_mean[0])])
    #         #     summary_5 = tf.Summary(value=[tf.Summary.Value(tag='A090', simple_value=acc_mean[1])])
    #         #     summary_6 = tf.Summary(value=[tf.Summary.Value(tag='A126', simple_value=acc_mean[2])])
    #     self.summary_writer.add_summary(summary_1, step)
    #     self.summary_writer.add_summary(summary_2, step)
    #     self.summary_writer.add_summary(summary_3, step)
    #     self.summary_writer.add_summary(summary_4, step)
    #     self.summary_writer.add_summary(summary_5, step)
    #     self.summary_writer.add_summary(summary_6, step)
    #     self.summary_writer.flush()

    def plot_acc(self, acc, sess):
        #     text = """ |         	| Probe 	|    	|    	|    	|      	|
        # \t|:-------:	|:-----:	|:--:	|:--:	|:--:	|:----:	|
        # \t| Gallery 	|   0   	| 30 	| 60 	| 90 	| Mean 	|
        # \t|    0    	|   %lf  	|  %lf 	|  %lf 	|  %lf 	|   %lf |
        # \t|    30   	|   %lf  	|  %lf 	|  %lf 	|  %lf 	|   %lf |
        # \t|    60   	|   %lf  	|  %lf 	|  %lf 	|  %lf 	|   %lf |
        # \t|    90   	|   %lf  	|  %lf 	|  %lf 	|  %lf 	|   %lf |
        # \t|   Mean  	|   %lf  	|  %lf 	|  %lf 	|  %lf 	|   %lf |""" % (
        #         acc[0, 0], acc[1, 0], acc[2, 0], acc[3, 0], np.mean(acc[:, 0]),
        #         acc[0, 1], acc[1, 1], acc[2, 1], acc[3, 1], np.mean(acc[:, 1]),
        #         acc[0, 2], acc[1, 2], acc[2, 2], acc[3, 2], np.mean(acc[:, 2]),
        #         acc[0, 3], acc[1, 3], acc[2, 3], acc[3, 3], np.mean(acc[:, 3]),
        #         np.mean(acc[0, :]), np.mean(acc[1, :]), np.mean(acc[2, :]), np.mean(acc[3, :], np.mean(acc)))
        tl = [
            ['', '0', '30', '60', '90'],
            ['0', str(acc[0, 0]), str(acc[1, 0]), str(acc[2, 0]), str(acc[3, 0])],
            ['30', str(acc[0, 1]), str(acc[1, 1]), str(acc[2, 1]), str(acc[3, 1])],
            ['60', str(acc[0, 2]), str(acc[1, 2]), str(acc[2, 2]), str(acc[3, 2])],
            ['90', str(acc[0, 3]), str(acc[1, 3]), str(acc[2, 3]), str(acc[3, 3])]
        ]
        summary_acc_op = tf.summary.text('acc_table', tf.convert_to_tensor(tl))
        text = sess.run(summary_acc_op)
        self.summary_writer.add_summary(text, 0)
        self.summary_writer.flush()
        return tl

    def form_acc_to_feed(self):
        acc_to_feed = []
        for i in range(4):
            acc_to_feed.append(self.acc[i, i])
        return np.asarray(acc_to_feed)


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    model = GaitGAN(tf_config=config)
    model.train(3)
