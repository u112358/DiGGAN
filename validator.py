import tensorflow as tf
import numpy as np
import tqdm
from sklearn.neighbors import NearestNeighbors

def validate(model, data_reader, sess):
    angles = [0, 30, 60, 90]
    acc = np.zeros((4, 4))
    with tqdm(total=16) as pbar:
        for i in range(4):
            for j in range(4):
                acc[i, j] = validate_probe_n_at_gallery_m(model, data_reader, sess, angles[i], angles[j])
                pbar.update(1)
    print_acc(acc)
    plot_acc(model.summary_writer, acc, sess)
    return acc


def validate_probe_n_at_gallery_m(model, data_reader, sess, n, m):
    list_probe_indices = data_reader.get_list_of_probe_at_angle_k(angle_k=n)
    len_of_list = list_probe_indices.shape[0]
    embeddings_probe = np.zeros((len_of_list, 128))
    idx_probe = []
    nof_batch_probe = int(np.floor(len_of_list / model.batch_size))
    for i in range(nof_batch_probe):
        x, _idx_probe, _angle_probe = data_reader.get_probe_by_indices(
            list_probe_indices[i * model.batch_size:(i + 1) * model.batch_size])
        x = np.expand_dims(np.array(x), -1)
        emb = sess.run(model.embeddings, feed_dict={model.input: x})
        embeddings_probe[i * model.batch_size:(i + 1) * model.batch_size, :] = emb
        idx_probe.append(_idx_probe)

    list_gallery_indices = data_reader.get_list_of_gallery_at_angle_k(angle_k=m)
    len_of_list = list_gallery_indices.shape[0]
    embeddings_gallery = np.zeros((len_of_list, 128))
    idx_gallery = []
    nof_batch_gallery = int(np.floor(len_of_list / model.batch_size))
    for i in range(nof_batch_gallery):
        x, _idx_gallery, _angle_gallery = data_reader.get_gallery_by_indices(
            list_gallery_indices[i * model.batch_size:(i + 1) * model.batch_size])
        x = np.expand_dims(np.array(x), -1)
        emb = sess.run(model.embeddings, feed_dict={model.input: x})
        embeddings_gallery[i * model.batch_size:(i + 1) * model.batch_size, :] = emb
        idx_gallery.append(_idx_gallery)
    #
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(embeddings_gallery[0:nof_batch_gallery * model.batch_size, :])
    distances, indices = nn.kneighbors(embeddings_probe[0:nof_batch_probe * model.batch_size, :])

    idx_gallery = np.reshape(np.asarray(idx_gallery), (-1, 1))
    idx_probe = np.reshape(np.asarray(idx_probe), (-1, 1))
    predicted = idx_gallery[indices, :]

    diff = np.reshape(predicted, (-1, 1)) - idx_probe[0:nof_batch_probe * model.batch_size, :]
    acc = np.where(diff == 0)[0].shape[0] / float(nof_batch_probe * model.batch_size)
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


def plot_acc(summary_writer, acc, sess):
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
    summary_writer.add_summary(text, 0)
    summary_writer.flush()


def print_acc(acc):
    print('\t\t0\t\t30\t\t60\t\t90\t')
    print('0\t%lf\t%lf\t%lf\t%lf\t' % (acc[0, 0], acc[1, 0], acc[2, 0], acc[3, 0]))
    print('30\t%lf\t%lf\t%lf\t%lf\t' % (acc[0, 1], acc[1, 1], acc[2, 1], acc[3, 1]))
    print('60\t%lf\t%lf\t%lf\t%lf\t' % (acc[0, 2], acc[1, 2], acc[2, 2], acc[3, 2]))
    print('90\t%lf\t%lf\t%lf\t%lf\t' % (acc[0, 3], acc[1, 3], acc[2, 3], acc[3, 3]))
