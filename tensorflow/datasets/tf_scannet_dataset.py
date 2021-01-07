import os
import sys
import numpy as np
import time
import pickle
from sklearn.neighbors import KDTree
import tensorflow as tf

# OS functions  我自己加的
from os import makedirs, listdir
from os.path import exists, join, isfile, isdir

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'Scannet')

if not os.path.exists(DATA_DIR):
    raise IOError(f"{DATA_DIR} not found!")

from utils.ply import read_ply, write_ply
from .custom_dataset import CustomDataset, grid_subsampling, tf_batch_subsampling, tf_batch_neighbors

class ScannetDataset(CustomDataset):
    
    def __init__(self, config, input_threads=8,load_test=False):
        """Class to handle S3DIS dataset for scene segmentation task.

        Args:
            config: config file
            input_threads: the number elements to process in parallel
        """
        super(ScannetDataset, self).__init__()
        self.config = config
        self.num_threads = input_threads

        # Dict from labels to names
        self.label_to_names = {0: 'unclassified',
                               1: 'wall',
                               2: 'floor',
                               3: 'cabinet',
                               4: 'bed',
                               5: 'chair',
                               6: 'sofa',
                               7: 'table',
                               8: 'door',
                               9: 'window',
                               10: 'bookshelf',
                               11: 'picture',
                               12: 'counter',
                               14: 'desk',
                               16: 'curtain',
                               24: 'refridgerator',
                               28: 'shower curtain',
                               33: 'toilet',
                               34: 'sink',
                               36: 'bathtub',
                               39: 'otherfurniture'}
        # Initiate a bunch of variables concerning class labels
        self.init_labels()
        # List of classes ignored during training (can be empty)
        self.ignored_labels = np.sort([0])
        ####################
        # Dataset parameters
        ####################
        # Type of task conducted on this dataset
        self.network_model = 'cloud_segmentation'
        # Number of input threads
        self.num_threads = input_threads
        ##########################
        # Parameters for the files
        ##########################
        # Path of the folder containing ply files
        self.path = 'data/Scannet'
        # Path of the training files
        self.train_path = join(self.path, 'training_points')
        self.test_path = join(self.path, 'test_points')
        # List of training and test files
        self.train_files = np.sort([join(self.train_path, f) for f in listdir(self.train_path) if f[-4:] == '.ply'])
        self.test_files = np.sort([join(self.test_path, f) for f in listdir(self.test_path) if f[-4:] == '.ply'])
        # Proportion of validation scenes
        self.validation_clouds = np.loadtxt(join(self.path, 'scannet_v2_val.txt'), dtype=np.str)
        # 1 to do validation, 2 to train on all data
        self.validation_split = 1
        self.all_splits = []
        # Load test set or train set?
        self.load_test = load_test

        # Some configs
        self.num_gpus = config.num_gpus
        #self.first_subsampling_dl = config.first_subsampling_dl
        self.in_features_dim = config.in_features_dim
        self.num_layers = config.num_layers
        self.downsample_times = config.num_layers - 1
        self.first_subsampling_dl = config.first_subsampling_dl
        self.density_parameter = config.density_parameter
        self.batch_size = config.batch_size
        self.augment_scale_anisotropic = config.augment_scale_anisotropic
        self.augment_symmetries = config.augment_symmetries
        self.augment_rotation = config.augment_rotation
        self.augment_scale_min = config.augment_scale_min
        self.augment_scale_max = config.augment_scale_max
        self.augment_noise = config.augment_noise
        self.augment_color = config.augment_color
        self.epoch_steps = config.epoch_steps
        self.validation_size = config.validation_size
        self.in_radius = config.in_radius

        # Prepare ply files
        #self.prepare_pointcloud_ply()

        # input subsampling
        self.load_subsampled_clouds(self.first_subsampling_dl)

        self.batch_limit = self.calibrate_batches()

        #self.neighborhood_limits = [26, 31, 38, 41, 39]
        self.neighborhood_limits=[34,30,35,37,35]
        self.neighborhood_limits = [int(l * self.density_parameter // 5) for l in self.neighborhood_limits]
        # hist_n = int(np.ceil(4 / 3 * np.pi * (config.density_parameter + 1) ** 3)) #根据配置参数，计算邻域中邻居数的上限
        # self.neighborhood_limits = np.full(config.num_layers, hist_n, dtype=np.int32)
        #print("neighborhood_limits: ", self.neighborhood_limits)

        # Get generator and mapping function
        gen_function, gen_types, gen_shapes = self.get_batch_gen('training')
        gen_function_val, _, _ = self.get_batch_gen('validation')
        map_func = self.get_tf_mapping()

        ##################
        # Training dataset
        ##################
        self.train_data = tf.data.Dataset.from_generator(gen_function,
                                                         gen_types,
                                                         gen_shapes)
        self.train_data = self.train_data.map(map_func=map_func, num_parallel_calls=self.num_threads)
        self.train_data = self.train_data.prefetch(10)
        ##############
        # Test dataset
        ##############
        # Create batched dataset from generator
        self.val_data = tf.data.Dataset.from_generator(gen_function_val,
                                                       gen_types,
                                                       gen_shapes)
        # Transform inputs
        self.val_data = self.val_data.map(map_func=map_func, num_parallel_calls=self.num_threads)
        # Prefetch data
        self.val_data = self.val_data.prefetch(10)

        #################
        # Common iterator
        #################

        # create a iterator of the correct shape and type
        iter = tf.data.Iterator.from_structure(self.train_data.output_types, self.train_data.output_shapes)
        self.flat_inputs = [None] * self.num_gpus
        for i in range(self.num_gpus):
            self.flat_inputs[i] = iter.get_next()
        # create the initialisation operations
        self.train_init_op = iter.make_initializer(self.train_data)
        self.val_init_op = iter.make_initializer(self.val_data)

        # input subsampling
    def load_subsampled_clouds(self, subsampling_parameter):
        """
        Presubsample point clouds and load into memory (Load KDTree for neighbors searches
        """

        if 0 < subsampling_parameter <= 0.01:
            raise ValueError('subsampling_parameter too low (should be over 1 cm')
            
        # Create path for files
        # print("self.path:",self.path)
        # assert 1==2
        tree_path = join(self.path, 'input_{:.3f}'.format(subsampling_parameter))
        if not exists(tree_path):
            makedirs(tree_path)

        # All training and test files
        files = np.hstack((self.train_files, self.test_files))

        # Initiate containers
        self.input_trees = {'training': [], 'validation': [], 'test': []}
        self.input_colors = {'training': [], 'validation': [], 'test': []}
        self.input_vert_inds = {'training': [], 'validation': [], 'test': []}
        self.input_labels = {'training': [], 'validation': []}

        # Advanced display
        N = len(files)
        # print("N:",N) #N: 1613
        # assert 1==2
        progress_n = 30
        fmt_str = '[{:<' + str(progress_n) + '}] {:5.1f}%'
        # print("fmt_str:",fmt_str)
        # assert 1==2
        print('\nPreparing KDTree for all scenes, subsampled at {:.3f}'.format(subsampling_parameter))

        # for i,file_path in enumerate(files):
        #     print("file_path:",file_path)
        # assert 1==2

        for i, file_path in enumerate(files):

            # Restart timer
            t0 = time.time()

            # get cloud name and split
            cloud_name = file_path.split('/')[-1][:-4]
            # print("cloud_name:",cloud_name)
            # assert 1==2
            cloud_folder = file_path.split('/')[-2]
            if 'train' in cloud_folder:
                if cloud_name in self.validation_clouds:
                    self.all_splits += [1]
                    cloud_split = 'validation'
                else:
                    self.all_splits += [0]
                    cloud_split = 'training'
            else:
                cloud_split = 'test'

            if (cloud_split != 'test' and self.load_test) or (cloud_split == 'test' and not self.load_test):
                continue

            # Name of the input files
            KDTree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            # Check if inputs have already been computed
            if isfile(KDTree_file):

                # read ply with data
                data = read_ply(sub_ply_file)
                #print("data['red'].shape:",data['red'].shape) #eg:(82,)
                sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
                sub_vert_inds = data['vert_ind']
                # print("sub_vert_inds:",sub_vert_inds)
                # assert 1==2
                if cloud_split == 'test':
                    sub_labels = None
                else:
                    sub_labels = data['class']
                    # print("sub_labels:",sub_labels)
                    # assert 1==2

                # Read pkl with search tree
                with open(KDTree_file, 'rb') as f:
                    search_tree = pickle.load(f)

            else:

                # Read ply file
                data = read_ply(file_path)
                points = np.vstack((data['x'], data['y'], data['z'])).T
                colors = np.vstack((data['red'], data['green'], data['blue'])).T
                if cloud_split == 'test':
                    int_features = data['vert_ind']
                else:
                    int_features = np.vstack((data['vert_ind'], data['class'])).T
                # Subsample cloud
                sub_points, sub_colors, sub_int_features = grid_subsampling(points,
                                                                      features=colors,
                                                                      labels=int_features,
                                                                      sampleDl=subsampling_parameter)
                # Rescale float color and squeeze label
                sub_colors = sub_colors / 255
                if cloud_split == 'test':
                    sub_vert_inds = np.squeeze(sub_int_features)
                    sub_labels = None
                else:
                    sub_vert_inds = sub_int_features[:, 0]
                    sub_labels = sub_int_features[:, 1]
                # Get chosen neighborhoods
                search_tree = KDTree(sub_points, leaf_size=50)
                # Save KDTree
                with open(KDTree_file, 'wb') as f:
                    pickle.dump(search_tree, f)
                # Save ply
                if cloud_split == 'test':
                    write_ply(sub_ply_file,
                              [sub_points, sub_colors, sub_vert_inds],
                              ['x', 'y', 'z', 'red', 'green', 'blue', 'vert_ind'])
                else:
                    write_ply(sub_ply_file,
                              [sub_points, sub_colors, sub_labels, sub_vert_inds],
                              ['x', 'y', 'z', 'red', 'green', 'blue', 'class', 'vert_ind'])

            # Fill data containers
            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            self.input_vert_inds[cloud_split] += [sub_vert_inds]
            if cloud_split in ['training', 'validation']:
                self.input_labels[cloud_split] += [sub_labels]

            print('', end='\r')
            print(fmt_str.format('#' * ((i * progress_n) // N), 100 * i / N), end='', flush=True)

        # Get number of clouds
        self.num_training = len(self.input_trees['training'])
        self.num_validation = len(self.input_trees['validation'])
        self.num_test = len(self.input_trees['test'])

        # Get validation and test reprojection indices
        self.validation_proj = []
        self.validation_labels = []
        self.test_proj = []
        self.test_labels = []
        i_val = 0
        i_test = 0

        # Advanced display
        N = self.num_validation + self.num_test
        print('', end='\r')
        print(fmt_str.format('#' * progress_n, 100), flush=True)
        print('\nPreparing reprojection indices for validation and test')

        for i, file_path in enumerate(files):
            # get cloud name and split
            cloud_name = file_path.split('/')[-1][:-4]
            cloud_folder = file_path.split('/')[-2]
            # Validation projection and labels
            if (not self.load_test) and 'train' in cloud_folder and cloud_name in self.validation_clouds:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))

                if isfile(proj_file):
                    with open(proj_file, 'rb') as f:
                        proj_inds, labels = pickle.load(f)
                else:
                    # Get original mesh
                    mesh_path = file_path.split('/') #['Data', 'Scannet', 'training_points', 'scene0000_00.ply']
                    # print("mesh_path:",mesh_path)
                    # assert 1==2
                    mesh_path[-2] = 'training_meshes'
                    mesh_path = '/'.join(mesh_path)
                    vertex_data, faces = read_ply(mesh_path[:-4] + '_mesh.ply', triangular_mesh=True)
                    vertices = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T
                    labels = vertex_data['class']
                    # Compute projection inds
                    proj_inds = np.squeeze(self.input_trees['validation'][i_val].query(vertices, return_distance=False))
                    proj_inds = proj_inds.astype(np.int32)
                    # Save
                    with open(proj_file, 'wb') as f:
                        pickle.dump([proj_inds, labels], f)
                self.validation_proj += [proj_inds]
                self.validation_labels += [labels]
                i_val += 1
            # Test projection
            if self.load_test and 'test' in cloud_folder:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                if isfile(proj_file):
                    with open(proj_file, 'rb') as f:
                        proj_inds, labels = pickle.load(f)
                else:
                    # Get original mesh
                    mesh_path = file_path.split('/')
                    mesh_path[-2] = 'test_meshes'
                    mesh_path = '/'.join(mesh_path)
                    vertex_data, faces = read_ply(mesh_path[:-4] + '_mesh.ply', triangular_mesh=True)
                    vertices = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T
                    labels = np.zeros(vertices.shape[0], dtype=np.int32)
                    # Compute projection inds
                    proj_inds = np.squeeze(self.input_trees['test'][i_test].query(vertices, return_distance=False))
                    proj_inds = proj_inds.astype(np.int32)

                    with open(proj_file, 'wb') as f:
                        pickle.dump([proj_inds, labels], f)
                self.test_proj += [proj_inds]
                self.test_labels += [labels]
                i_test += 1
            print('', end='\r')
            print(fmt_str.format('#' * (((i_val + i_test) * progress_n) // N), 100 * (i_val + i_test) / N),
                  end='',
                  flush=True)

        print('\n')

        return
    
    def calibrate_batches(self):
        if len(self.input_trees['training']) > 0:
            split = 'training'
        else:
            split = 'test'
        N = (10000 // len(self.input_trees[split])) + 1
        sizes = []

        # Take a bunch of example neighborhoods in all clouds
        for i, tree in enumerate(self.input_trees[split]):
            # Randomly pick points
            points = np.array(tree.data, copy=False) #吗每组里面有很多行，每行三个数是该点的xyz坐标
            rand_inds = np.random.choice(points.shape[0], size=N, replace=False) #从一个area中选取2001个点
            rand_points = points[rand_inds]
            noise = np.random.normal(scale=self.in_radius/4, size=rand_points.shape)
            rand_points += noise.astype(rand_points.dtype)
            neighbors = tree.query_radius(points[rand_inds], r=self.in_radius) #Query for neighbors within a given radius
            # Only save neighbors lengths
            sizes += [len(neighb) for neighb in neighbors]
        sizes = np.sort(sizes)
        # Higher bound for batch limit
        lim = sizes[-1] * self.batch_size
        # Biggest batch size with this limit
        sum_s = 0
        max_b = 0
        for i, s in enumerate(sizes):
            sum_s += s
            #print("sum_s:",sum_s) #从8,27...到155454
            if sum_s > lim:
                max_b = i
                #print("max_b:",max_b) #max_b: 97
                break
        # With a proportional corrector, find batch limit which gets the wanted batch_size
        estim_b = 0
        for i in range(10000):
            # Compute a random batch
            rand_shapes = np.random.choice(sizes, size=max_b, replace=False) ## 参数意思分别 是从sizes中以概率P，随机选择max_b个, p没有指定的时候相当于是一致的分布
            b = np.sum(np.cumsum(rand_shapes) < lim)
            # Update estim_b (low pass filter istead of real mean
            estim_b += (b - estim_b) / min(i+1, 100)
            # Correct batch limit
            lim += 10.0 * (self.batch_size - estim_b)
        return lim
    
    def get_batch_gen(self, split):
        """
        A function defining the batch generator for each split. Should return the generator, the generated types and
        generated shapes
        :param split: string in "training", "validation" or "test"
        :param config: configuration file
        :return: gen_func, gen_types, gen_shapes
        """
        ############
        # Parameters
        ############
        # Initiate parameters depending on the chosen split
        if split == 'training':
            # First compute the number of point we want to pick in each cloud and for each class
            epoch_n = self.epoch_steps * self.batch_size
            random_pick_n = None
        elif split == 'validation':
            # First compute the number of point we want to pick in each cloud and for each class
            epoch_n = self.validation_size * self.batch_size
        elif split == 'test':
            # First compute the number of point we want to pick in each cloud and for each class
            epoch_n = self.validation_size * self.batch_size
        elif split == 'ERF':
            # First compute the number of point we want to pick in each cloud and for each class
            epoch_n = 1000000
            self.batch_limit = 1
            np.random.seed(42)
        else:
            raise ValueError('Split argument in data generator should be "training", "validation" or "test"')

        # Initiate potentials for regular generation
        if not hasattr(self, 'potentials'):
            self.potentials = {}
            self.min_potentials = {}
        # Reset potentials
        self.potentials[split] = []
        self.min_potentials[split] = []
        data_split = split
        if split == 'ERF':
            data_split = 'test'
        for i, tree in enumerate(self.input_trees[data_split]):
            self.potentials[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_potentials[split] += [float(np.min(self.potentials[split][-1]))]
        ##########################
        # Def generators functions
        ##########################
        def spatially_regular_gen():
            # Initiate concatanation lists
            p_list = []
            c_list = []
            pl_list = []
            pi_list = []
            ci_list = []
            batch_n = 0
            # Generator loop
            for i in range(epoch_n):
                # Choose a random cloud
                cloud_ind = int(np.argmin(self.min_potentials[split]))
                # Choose point ind as minimum of potentials
                point_ind = np.argmin(self.potentials[split][cloud_ind])
                # Get points from tree structure
                points = np.array(self.input_trees[data_split][cloud_ind].data, copy=False)
                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)
                # Add noise to the center point
                if split != 'ERF':
                    noise = np.random.normal(scale=self.in_radius/10, size=center_point.shape)
                    pick_point = center_point + noise.astype(center_point.dtype)
                else:
                    pick_point = center_point
                # Indices of points in input region
                input_inds = self.input_trees[data_split][cloud_ind].query_radius(pick_point,
                                                                                  r=self.in_radius)[0]
                # Number collected
                n = input_inds.shape[0]
                # Update potentials (Tuckey weights)
                if split != 'ERF':
                    dists = np.sum(np.square((points[input_inds] - pick_point).astype(np.float32)), axis=1)
                    tukeys = np.square(1 - dists / np.square(self.in_radius))
                    tukeys[dists > np.square(self.in_radius)] = 0
                    self.potentials[split][cloud_ind][input_inds] += tukeys
                    self.min_potentials[split][cloud_ind] = float(np.min(self.potentials[split][cloud_ind]))
                    # Safe check for very dense areas
                    if n > self.batch_limit:
                        input_inds = np.random.choice(input_inds, size=int(self.batch_limit)-1, replace=False)
                        n = input_inds.shape[0]
                # Collect points and colors
                input_points = (points[input_inds] - pick_point).astype(np.float32)
                input_colors = self.input_colors[data_split][cloud_ind][input_inds]
                if split in ['test', 'ERF']:
                    input_labels = np.zeros(input_points.shape[0])
                else:
                    input_labels = self.input_labels[data_split][cloud_ind][input_inds]
                    input_labels = np.array([self.label_to_idx[l] for l in input_labels])
                # In case batch is full, yield it and reset it
                if batch_n + n > self.batch_limit and batch_n > 0:
                    yield (np.concatenate(p_list, axis=0),
                           np.concatenate(c_list, axis=0),
                           np.concatenate(pl_list, axis=0),
                           np.array([tp.shape[0] for tp in p_list]),
                           np.concatenate(pi_list, axis=0),
                           np.array(ci_list, dtype=np.int32))
                    p_list = []
                    c_list = []
                    pl_list = []
                    pi_list = []
                    ci_list = []
                    batch_n = 0
                # Add data to current batch
                if n > 0:
                    p_list += [input_points]
                    c_list += [np.hstack((input_colors, input_points + pick_point))]
                    pl_list += [input_labels]
                    pi_list += [input_inds]
                    ci_list += [cloud_ind]
                # Update batch size
                batch_n += n
            if batch_n > 0:
                yield (np.concatenate(p_list, axis=0),
                       np.concatenate(c_list, axis=0),
                       np.concatenate(pl_list, axis=0),
                       np.array([tp.shape[0] for tp in p_list]),
                       np.concatenate(pi_list, axis=0),
                       np.array(ci_list, dtype=np.int32))
        ###################
        # Choose generators
        ###################
        # Define the generator that should be used for this split
        if split == 'training':
            gen_func = spatially_regular_gen

        elif split == 'validation':
            gen_func = spatially_regular_gen

        elif split in ['test', 'ERF']:
            gen_func = spatially_regular_gen

        else:
            raise ValueError('Split argument in data generator should be "training", "validation" or "test"')

        # Define generated types and shapes
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.int32)
        gen_shapes = ([None, 3], [None, 6], [None], [None], [None], [None])

        return gen_func, gen_types, gen_shapes
    
    def get_tf_mapping(self):

        # Returned mapping function
        def tf_map(stacked_points, stacked_colors, point_labels, stacks_lengths, point_inds, cloud_inds):
            
            # Get batch indice for each point
            batch_inds = self.tf_get_batch_inds(stacks_lengths)

            # Augment input points
            stacked_points, scales, rots = self.tf_augment_input(stacked_points,
                                                                 batch_inds)
            
            # First add a column of 1 as feature for the network to be able to learn 3D shapes
            stacked_features = tf.ones((tf.shape(stacked_points)[0], 1), dtype=tf.float32)

            # Get coordinates and colors
            stacked_original_coordinates = stacked_colors[:, 3:]
            stacked_colors = stacked_colors[:, :3]

            # Augmentation : randomly drop colors
            if self.in_features_dim in [4, 5]:
                num_batches = batch_inds[-1] + 1
                s = tf.cast(tf.less(tf.random_uniform((num_batches,)), self.augment_color), tf.float32)
                stacked_s = tf.gather(s, batch_inds)
                stacked_colors = stacked_colors * tf.expand_dims(stacked_s, axis=1)
            # Then use positions or not
            if self.in_features_dim == 1:
                pass
            elif self.in_features_dim == 2:
                stacked_features = tf.concat((stacked_features, stacked_original_coordinates[:, 2:]), axis=1)
            elif self.in_features_dim == 3:
                stacked_features = stacked_colors
            elif self.in_features_dim == 4:
                stacked_features = tf.concat((stacked_features, stacked_colors), axis=1)
            elif self.in_features_dim == 5:
                stacked_features = tf.concat((stacked_features, stacked_colors, stacked_original_coordinates[:, 2:]), axis=1)
            elif self.in_features_dim == 7:
                stacked_features = tf.concat((stacked_features, stacked_colors, stacked_points), axis=1)
            else:
                raise ValueError('Only accepted input dimensions are 1, 3, 4 and 7 (without and with rgb/xyz)')

            # Get the whole input list
            input_list = self.tf_segmentation_inputs(self.downsample_times,
                                                     self.first_subsampling_dl,
                                                     self.density_parameter,
                                                     stacked_points,
                                                     stacked_features,
                                                     point_labels,
                                                     stacks_lengths,
                                                     batch_inds)
            # Add scale and rotation for testing
            input_list += [scales, rots]
            input_list += [point_inds, cloud_inds]

            return input_list

        return tf_map

    def load_evaluation_points(self, file_path):
        """
        Load points (from test or validation split) on which the metrics should be evaluated
        """

        # Get original points
        data = read_ply(file_path)
        return np.vstack((data['x'], data['y'], data['z'])).T
    
    
    def tf_segmentation_inputs(self,
                               downsample_times,
                               first_subsampling_dl,
                               density_parameter,
                               stacked_points,
                               stacked_features,
                               point_labels,
                               stacks_lengths,
                               batch_inds):
        # Batch weight at each point for loss (inverse of stacks_lengths for each point)
        min_len = tf.reduce_min(stacks_lengths, keep_dims=True)
        batch_weights = tf.cast(min_len, tf.float32) / tf.cast(stacks_lengths, tf.float32)
        stacked_weights = tf.gather(batch_weights, batch_inds)
        # Starting radius of convolutions
        dl = first_subsampling_dl
        dp = density_parameter
        r = dl * dp / 2.0
        # Lists of inputs
        num_layers = (downsample_times + 1)
        input_points = [None] * num_layers
        input_neighbors = [None] * num_layers
        input_pools = [None] * num_layers
        input_upsamples = [None] * num_layers
        input_batches_len = [None] * num_layers

        neighbors_inds = tf_batch_neighbors(stacked_points, stacked_points, stacks_lengths, stacks_lengths, r)
        pool_points, pool_stacks_lengths = tf_batch_subsampling(stacked_points, stacks_lengths, sampleDl=2 * dl)
        pool_inds = tf_batch_neighbors(pool_points, stacked_points, pool_stacks_lengths, stacks_lengths, r)
        up_inds = tf_batch_neighbors(stacked_points, pool_points, stacks_lengths, pool_stacks_lengths, 2 * r)
        neighbors_inds = self.big_neighborhood_filter(neighbors_inds, 0)
        pool_inds = self.big_neighborhood_filter(pool_inds, 0)
        up_inds = self.big_neighborhood_filter(up_inds, 0)
        input_points[0] = stacked_points
        input_neighbors[0] = neighbors_inds
        input_pools[0] = pool_inds
        input_upsamples[0] = tf.zeros((0, 1), dtype=tf.int32)
        input_upsamples[1] = up_inds
        input_batches_len[0] = stacks_lengths
        stacked_points = pool_points
        stacks_lengths = pool_stacks_lengths
        r *= 2
        dl *= 2

        for dt in range(1, downsample_times):
            neighbors_inds = tf_batch_neighbors(stacked_points, stacked_points, stacks_lengths, stacks_lengths, r)
            pool_points, pool_stacks_lengths = tf_batch_subsampling(stacked_points, stacks_lengths, sampleDl=2 * dl)
            pool_inds = tf_batch_neighbors(pool_points, stacked_points, pool_stacks_lengths, stacks_lengths, r)
            up_inds = tf_batch_neighbors(stacked_points, pool_points, stacks_lengths, pool_stacks_lengths, 2 * r)
            neighbors_inds = self.big_neighborhood_filter(neighbors_inds, dt)
            pool_inds = self.big_neighborhood_filter(pool_inds, dt)
            up_inds = self.big_neighborhood_filter(up_inds, dt)
            input_points[dt] = stacked_points
            input_neighbors[dt] = neighbors_inds
            input_pools[dt] = pool_inds
            input_upsamples[dt + 1] = up_inds
            input_batches_len[dt] = stacks_lengths
            stacked_points = pool_points
            stacks_lengths = pool_stacks_lengths
            r *= 2
            dl *= 2

        neighbors_inds = tf_batch_neighbors(stacked_points, stacked_points, stacks_lengths, stacks_lengths, r)
        neighbors_inds = self.big_neighborhood_filter(neighbors_inds, downsample_times)
        input_points[downsample_times] = stacked_points
        input_neighbors[downsample_times] = neighbors_inds
        input_pools[downsample_times] = tf.zeros((0, 1), dtype=tf.int32)
        input_batches_len[downsample_times] = stacks_lengths

        # Batch unstacking (with first layer indices for optionnal classif loss)
        stacked_batch_inds_0 = self.tf_stack_batch_inds(input_batches_len[0])
        # Batch unstacking (with last layer indices for optionnal classif loss)
        stacked_batch_inds_1 = self.tf_stack_batch_inds(input_batches_len[-1])
        # list of network inputs
        # list of network inputs
        li = input_points + input_neighbors + input_pools + input_upsamples
        li += [stacked_features, stacked_weights, stacked_batch_inds_0, stacked_batch_inds_1]
        li += [point_labels]
        return li