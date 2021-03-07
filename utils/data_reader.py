from os import listdir
from os.path import isfile, join, exists

import numpy as np
# from PIL import Image
from scipy.misc import imread
import random
import time


class DataReader:
    # root_path: path_to_'/GEI' in your file system
    # for example, your dataset has a structure like /Downloads/GEI/000-00/00001.png
    # the root_path here should be root_path='/Downloads/GEI'
    def __init__(self, root_path, usage='default usage'):
        start_time = time.time()
        self.__usage = usage
        print('Initialising data reader for ' + self.__usage + '...')

        print('Spliting training and testing set ...')
        self.root_path = root_path
        self.num_views = 14
        self.dirs_in_root_path = listdir(self.root_path)

        # self.***_data_path is a list of the GEIs' paths
        self.training_data_path = []
        self.testing_data_probe_path = []
        self.testing_data_gallery_path = []

        # self.***_data_id is a list of the GEIs' subject_idx
        self.training_data_id = []
        self.testing_data_probe_id = []
        self.testing_data_gallery_id = []

        # self.***_data_angle is a list of the GEIs' angle
        self.training_data_angle = []
        self.testing_data_probe_angle = []
        self.testing_data_gallery_angle = []

        for subdir in self.dirs_in_root_path:
            file_list = listdir(join(self.root_path, subdir))
            # All the ODD indexed subjects in both 00 and 01 folders are used for training
            # The EVEN indexed subjects are formed into probe, if the GEIs are in folders ending with '00';
            # and are formed into gallery, if the GEIs are in folders ending with '01'
            for file_name in file_list:
                full_file_path = join(self.root_path, subdir, file_name)
                if isfile(full_file_path) and int(file_name.replace('.png', '')) % 2 == 1:
                    self.training_data_path.append(full_file_path)
                    self.training_data_id.append(int(file_name.replace('.png', '')))
                    self.training_data_angle.append(int(subdir[0:3]))
                else:
                    if subdir[-1] == '0':
                        self.testing_data_probe_path.append(full_file_path)
                        self.testing_data_probe_id.append(int(file_name.replace('.png', '')))
                        self.testing_data_probe_angle.append(int(subdir[0:3]))
                    else:
                        self.testing_data_gallery_path.append(full_file_path)
                        self.testing_data_gallery_id.append(int(file_name.replace('.png', '')))
                        self.testing_data_gallery_angle.append(int(subdir[0:3]))

        self.size_of_training_data = len(self.training_data_path)
        self.size_of_probe_data = len(self.testing_data_probe_path)
        self.size_of_gallery_data = len(self.testing_data_gallery_path)

        print('#Traning samples: %d \t #Probe samples: %d \t #Gallery samples: %d ' % (
            self.size_of_training_data, self.size_of_probe_data, self.size_of_gallery_data))

        print('Shuffling training set ...')
        self.idx_list_training = np.arange(self.size_of_training_data)
        np.random.shuffle(self.idx_list_training)
        self.current_idx_training = 0

        self.id_list = np.unique(self.training_data_id)
        self.label_mapping = {}
        for i in range(len(self.id_list)):
            self.label_mapping[self.id_list[i]] = i
        print('Done!\n Time used:%s seconds' % str(time.time() - start_time))

    def __del__(self):
        print('Releasing data reader for ' + self.__usage + '...')
        print('Done!\n')

    def get_next_batch_with_random_angle(self, batch_size):
        x = []
        angle = []
        if self.current_idx_training + batch_size > self.size_of_training_data:
            self.current_idx_training = 0
            np.random.shuffle(self.idx_list_training)
        batch_list = self.idx_list_training[self.current_idx_training:self.current_idx_training + batch_size]
        self.current_idx_training += batch_size
        for b in batch_list:
            filename = self.training_data_path[b]
            _x = imread(filename)
            _x = np.asarray(_x) / 255.0
            x.append(_x)
            if int(filename[-16:-13]) > 90:
                _idx = int(filename[-16:-13]) / 15 - 5
            else:
                _idx = int(filename[-16:-13]) / 15
            _angle = np.zeros(self.num_views)
            _angle[int(_idx)] = 1
            angle.append(_angle)
        return x, angle

    def get_next_batch_at_given_angle(self, batch_size, angle_list):
        # angle is in [0,1,...,13] for 14 view angles
        x = []
        paired_x = []
        count = 0
        while count < batch_size:
            if self.current_idx_training + 1 > self.size_of_training_data:
                self.current_idx_training = 0
                np.random.shuffle(self.idx_list_training)
            idx = self.idx_list_training[self.current_idx_training]
            self.current_idx_training += 1
            filename = self.training_data_path[idx]
            id = "%05d" % self.training_data_id[idx]
            angle = angle_list[count]
            paired_filename = self.get_path_by_id_and_angle(id, angle)
            if not isfile(paired_filename):
                continue
            _x = imread(filename)
            _x = np.asarray(_x) / 255.0
            _paired_x = imread(paired_filename)
            _paired_x = np.asarray(_paired_x) / 255.0
            x.append(_x)
            paired_x.append(_paired_x)
            count += 1
        # if self.current_idx_training + batch_size > self.size_of_training_data:
        #     self.current_idx_training = 0
        #     np.random.shuffle(self.idx_list_training)
        # batch_list = self.idx_list_training[self.current_idx_training:self.current_idx_training + batch_size]
        # self.current_idx_training += batch_size
        # for i, b in enumerate(batch_list):
        #     filename = self.training_data_path[b]
        #     id = "%05d" % self.training_data_id[b]
        #     angle = angle_list[i]
        #     paired_filename = self.get_path_by_id_and_angle(id, angle)
        #     if not isfile(paired_filename):
        #         print(paired_filename)
        #         filename = join(self.root_path,'000-00/00001.png')
        #         paired_filename = join(self.root_path,'090-00/00001.png')
        #         print('file does not exist exception resolved!')
        #     _x = imread(filename)
        #     _x = np.asarray(_x) / 255.0
        #     _paired_x = imread(paired_filename)
        #     _paired_x = np.asarray(_paired_x) / 255.0
        #     x.append(_x)
        #     paired_x.append(_paired_x)
        return x, paired_x

    def select_id(self, nof_person, nof_images):
        x = []
        label = []
        paths = []
        k = nof_person
        ids_selected = random.sample(list(self.id_list), nof_person)
        for i in ids_selected:
            image_indices = np.where(self.training_data_id == i)[0]
            if len(image_indices) >= nof_images:
                images_selected = random.sample(range(len(image_indices)), nof_images)
                images_selected = image_indices[images_selected]
            else:
                k -= 1
                continue

            for image in images_selected:
                path = self.training_data_path[image]
                paths.append(path)
                _x = imread(path)
                _x = np.array(_x) / 255.0
                x.append(_x)
                label.append(i)
        return x, label, np.asarray(paths), k

    def select_id_v2(self, nof_person, nof_images):
        x = []
        label = []
        paths = []
        k = nof_person
        angle_0_list = np.where(np.asarray(self.training_data_angle) == 0)[0]
        angle_30_list = np.where(np.asarray(self.training_data_angle) == 30)[0]
        angle_60_list = np.where(np.asarray(self.training_data_angle) == 60)[0]
        angle_90_list = np.where(np.asarray(self.training_data_angle) == 90)[0]
        small_list = np.concatenate([angle_0_list, angle_30_list, angle_60_list, angle_90_list])
        small_training_data_id = np.asarray(self.training_data_id)[small_list]
        small_training_data_path = np.asarray(self.training_data_path)[small_list]
        ids_selected = random.sample(list(self.id_list), nof_person)
        for i in ids_selected:
            image_indices = np.where(small_training_data_id == i)[0]
            if len(image_indices) >= nof_images:
                images_selected = random.sample(range(len(image_indices)), nof_images)
                images_selected = image_indices[images_selected]
            else:
                k -= 1
                continue
            for image in images_selected:
                path = small_training_data_path[image]
                paths.append(path)
                _x = imread(path)
                _x = np.array(_x) / 255.0
                x.append(_x)
                label.append(i)
        return x, label, np.asarray(paths), k

    def read_from_paths(self, path_list):
        x = []
        for path in path_list:
            _x = imread(path)
            _x = np.array(_x) / 255.0
            x.append(_x)
        return x

    def get_probe_by_indices(self, list_of_indices):
        x = []
        idx = []
        angle = []
        for _index in list_of_indices:
            path = self.testing_data_probe_path[_index]
            _x = imread(path)
            _x = np.array(_x) / 255.0
            x.append(_x)
            idx.append(self.testing_data_probe_id[_index])
            angle.append(self.testing_data_probe_angle[_index])
        return x, idx, angle

    def get_gallery_by_indices(self, list_of_indices):
        x = []
        idx = []
        angle = []
        for _index in list_of_indices:
            path = self.testing_data_gallery_path[_index]
            _x = imread(path)
            _x = np.array(_x) / 255.0
            x.append(_x)
            idx.append(self.testing_data_gallery_id[_index])
            angle.append(self.testing_data_gallery_angle[_index])
        return x, idx, angle

    def get_list_of_probe_at_angle_k(self, angle_k):
        list_of_indices = np.where(np.asarray(self.testing_data_probe_angle) == angle_k)[0]
        return list_of_indices

    def get_list_of_gallery_at_angle_k(self, angle_k):
        list_of_indices = np.where(np.asarray(self.testing_data_gallery_angle) == angle_k)[0]
        return list_of_indices

    def get_path_by_id_and_angle(self, id, angle):
        angle_set = ['000', '015', '030', '045', '060', '075', '090', '180', '195', '210', '225', '240', '255', '270']
        path = join(self.root_path, angle_set[int(angle)] + '-00', str(id).zfill(5) + '.png')
        return path

    def select_pairs_for_contrastive_loss(self, batch_size):
        labels = np.zeros(batch_size, dtype=np.float)
        anchor_images = []
        positive_images = []
        for i in range(batch_size):
            anchor_index = random.sample(self.idx_list_training.tolist(), 1)[0]
            anchor_id = self.training_data_id[anchor_index]
            if i < batch_size / 2:
                candidates = np.where(np.asarray(self.training_data_id) == anchor_id)[0]
                positive_index = random.sample(list(candidates), 1)[0]
                labels[i] = 1.0
            else:
                candidates = np.where(np.asarray(self.training_data_id) != anchor_id)[0]
                positive_index = random.sample(list(candidates), 1)[0]
                labels[i] = 0.0
            _anchor_image = imread(self.training_data_path[anchor_index])
            _anchor_image = np.array(_anchor_image) / 255.0
            anchor_images.append(_anchor_image)
            _positive_image = imread(self.training_data_path[positive_index])
            _positive_image = np.array(_positive_image) / 255.0
            positive_images.append(_positive_image)
        return anchor_images, positive_images, labels

    def select_pairs_for_contrastive_loss_v2(self, batch_size):
        labels = np.zeros(batch_size, dtype=np.float)
        anchor_images = []
        positive_images = []
        angle_0_list = np.where(np.asarray(self.training_data_angle) == 0)[0]
        angle_30_list = np.where(np.asarray(self.training_data_angle) == 30)[0]
        angle_60_list = np.where(np.asarray(self.training_data_angle) == 60)[0]
        angle_90_list = np.where(np.asarray(self.training_data_angle) == 90)[0]
        small_list = np.concatenate([angle_0_list, angle_30_list, angle_60_list, angle_90_list])
        small_idx_list_training = np.arange(len(small_list))
        small_training_data_id = np.asarray(self.training_data_id)[small_list]
        small_training_data_path = np.asarray(self.training_data_path)[small_list]
        for i in range(batch_size):
            anchor_index = random.sample(list(small_idx_list_training), 1)[0]
            anchor_id = small_training_data_id[anchor_index]
            if i < batch_size / 2:
                candidates = np.where(np.asarray(small_training_data_id) == anchor_id)[0]
                positive_index = random.sample(list(candidates), 1)[0]
                labels[i] = 1.0
            else:
                candidates = np.where(np.asarray(small_training_data_id) != anchor_id)[0]
                positive_index = random.sample(list(candidates), 1)[0]
                labels[i] = 0.0
            _anchor_image = imread(small_training_data_path[anchor_index])
            _anchor_image = np.array(_anchor_image) / 255.0
            anchor_images.append(_anchor_image)
            _positive_image = imread(small_training_data_path[positive_index])
            _positive_image = np.array(_positive_image) / 255.0
            positive_images.append(_positive_image)
        return anchor_images, positive_images, labels

    def get_test_input(self, angle):
        path = self.get_path_by_id_and_angle(1992, angle)
        _image = imread(path)
        _image = np.asarray(_image) / 255.0
        return _image

    def select_data(self, nof_person, nof_images):
        x = []
        label = []
        paths = []
        batch_size = nof_images * nof_person
        k = nof_person
        angle_0_list = np.where(np.asarray(self.training_data_angle) == 0)[0]
        angle_30_list = np.where(np.asarray(self.training_data_angle) == 30)[0]
        angle_60_list = np.where(np.asarray(self.training_data_angle) == 60)[0]
        angle_90_list = np.where(np.asarray(self.training_data_angle) == 90)[0]
        small_list = np.concatenate([angle_0_list, angle_30_list, angle_60_list, angle_90_list])
        small_training_data_id = np.asarray(self.training_data_id)[small_list]
        small_training_data_path = np.asarray(self.training_data_path)[small_list]
        full = False
        while not full:
            ids_selected = random.sample(list(self.id_list), nof_person)
            for i in ids_selected:
                image_indices = np.where(small_training_data_id == i)[0]
                if len(image_indices) >= nof_images:
                    full = False
                    break
            full = True
        for i in ids_selected:
            images_selected = random.sample(range(len(image_indices)), nof_images)
            images_selected = image_indices[images_selected]
            for image in images_selected:
                path = small_training_data_path[image]
                paths.append(path)
                _x = imread(path)
                _x = np.array(_x) / 255.0
                x.append(_x)
                label.append(i)
        y = []
        for i in label:
            y.append(self.label_mapping[i])
        A = np.zeros([batch_size, batch_size])
        for i in range(batch_size):
            for j in range(batch_size):
                if y[i] == y[j]:
                    A[i, j] = 1

        return x, y, np.asarray(paths), A


if __name__ == '__main__':
    dr = DataReader(root_path='../GEI/GEI', usage='Training')
    dr.select_data(6, 5)
