"""
Caltech FlyvFly Dataset for preparing to be used by Neon.
More info at: http://www.vision.caltech.edu/Video_Datasets/Fly-vs-Fly/download.html
"""

import logging
import numpy as np, h5py 
import scipy.io

from neon.datasets.dataset import Dataset
import cProfile

MOVIE_DIR = "/home/coreyesj/flyvflydata/Aggression/Aggression"

# Feature constants
NUM_BINS = 20
NUM_FRAMES = 3
FEATURE_LENGTH = 2 * 17 * NUM_FRAMES * NUM_BINS

movie_nos = [1, 2, 3, 4, 5, 6, 7] # Not zero index
train_nos = [0, 1, 2, 3, 4] # Zero indexed
validation_nos = [5]
test_nos = [6]

logger = logging.getLogger(__name__)

def read_tracking_data(movie_no):
    """ Read tracking data from matlab structs"""

    matpath = "%s/movie%d/movie%d_track.mat" %(MOVIE_DIR, movie_no, movie_no)
    matfile = scipy.io.loadmat(matpath, struct_as_record=True)
    trk_flagframes, trk_names, trk_data = matfile['trk'][0,0]

    return (trk_flagframes, trk_names, trk_data)

def read_labels(movie_no):
    """ Read action labels from matlab structs"""

    matpath = "%s/movie%d/movie%d_actions.mat" %(MOVIE_DIR, movie_no, movie_no)
    matfile = scipy.io.loadmat(matpath,squeeze_me=False)
    action_names, action_labels = matfile['behs'], matfile['bouts']

    return (action_names, action_labels)

def discretize(trk_data, min_range, max_range, num_bins=NUM_BINS):
    """ Discretize tracking data into binary bins"""

    num_frames, num_features = trk_data.shape[1:3]

    # Create bins for each feature based on max and min ranges
    feature_bins = np.zeros((num_features, num_bins))
    for i in xrange(num_features):
        feature_bins[i, :] = np.linspace(min_range[i], max_range[i], num=num_bins)

    # Get bin number that each feature falls in
    #disc_trk_data = np.zeros(trk_data.shape)
    disc_trk_data = np.zeros((2, num_frames, num_features))
    binary_offsets = np.arange(0, num_bins*num_frames, num_bins)

    for j in xrange(num_features):
        # bin_nos will range from 0 to num_bins and be 0 or num_bins if outside range
        for fly_no in range(2):
            bin_nos = np.searchsorted(feature_bins[j,:], trk_data[fly_no,:, j])
            #binary_index = bin_nos + binary_offsets
            #binary_index = binary_index[~np.isnan(binary_index)]
            disc_trk_data[fly_no, :, j] = bin_nos

    # Keep nan's in trk_data as nans in disc
    disc_trk_data[np.isnan(trk_data)] = np.nan

    return disc_trk_data


def find_ranges():
    """ Find min and max ranges for features in all movies"""
    all_trk_data = read_tracking_data(movie_nos[0])[2]

    for movie_no in movie_nos[1:]:
        trk_data = read_tracking_data(movie_no)[2]
        all_trk_data = np.vstack((all_trk_data, trk_data))

    max_range = np.nanmax(all_trk_data, axis=1).max(axis=0)
    max_range *= (1+0.01*np.sign(max_range))
    min_range = np.nanmin(all_trk_data, axis=1).min(axis=0)
    min_range *= (1-0.01*np.sign(min_range))

    return min_range, max_range

def transform(trk_data, labels, window_length=3, stride=1):
    """Get sliding window of frames from tracking data"""

    num_frames, num_features = trk_data.shape[1:3]
    action_no = 0
    fly_no = 1
    feature_length = 2 * num_features*window_length;
    X = np.zeros((num_frames-window_length+1, feature_length))
    Y = np.zeros((X.shape[0], 1))
    relative_bin_offset = np.arange(0, feature_length*NUM_BINS, NUM_BINS)
    # Get window of tracking data over time
    for i in xrange(0, num_frames - window_length):
        window = trk_data[:, i:(i+window_length), :]
        X[i, :] = np.reshape(window, (1, window.size))
        # Change bin binary data to relative position of 1
        X[i, :] += relative_bin_offset
    action_labels = labels[fly_no, action_no][:, 0:2]
    for i in xrange(0, len(action_labels)):
        start, stop = action_labels[i, :]
        Y[start:stop, 0] = 1

    return X, Y

def filter(X, Y):
    """Filter out percentage of data"""
    idx1 = np.where(Y == 0)
    idx2 = np.where(Y == 1)
    no_action = int(0.2 * idx.shape[0])
    idx1 = idx1[0:no_action]
    num_points = len(idx1) + len(idx2)
    newX = np.zeros(())


def load_data():
    min_range, max_range = find_ranges()
    data = []
    for movie_no in movie_nos:
        trk_data = discretize(read_tracking_data(movie_no)[2], min_range, max_range)
        #trk_data = read_tracking_data(movie_no)[2]
        labels = read_labels(movie_no)[1]
        data.append(transform(trk_data, labels))

    return data


class Fly(Dataset):
    """
    Sets up the fly v fly dataset.

    Attributes:
        backend (neon.backends.Backend): backend used for this data
        inputs (dict): structure housing the loaded train/test/validation
                       input data
        targets (dict): structure housing the loaded train/test/validation
                        target data

    Kwargs:
        repo_path (str, optional): where to locally host this dataset on disk
    """

    def __init__(self, **kwargs):
        self.macro_batched = False
        self.dist_flag = False
        self.num_test_sample = 10000
        self.__dict__.update(kwargs)
        if self.dist_flag:
            raise NotImplementedError("Dist not yet implemented for Chess")
        # if not hasattr(self, 'save_dir'):
        #     self.save_dir = os.path.join(self.repo_path,
        #                                  self.__class__.__name__)

    def load(self):
        if self.inputs['train'] is not None:
            return

        flydata = load_data()
        self.inputs['train'] = np.vstack([flydata[i][0] for i in train_nos])
        self.targets['train'] = np.vstack([flydata[i][1] for i in train_nos])
        print "Training size: ", self.inputs['train'].shape
        self.inputs['validation'] = np.vstack([flydata[i][0] for i in validation_nos])
        self.targets['validation'] = np.vstack([flydata[i][1] for i in validation_nos])  
        self.inputs['test'] = np.vstack([flydata[i][0] for i in test_nos])
        self.targets['test'] = np.vstack([flydata[i][1] for i in test_nos])
        print "Test Size: ", self.inputs['test'].shape
        self.format()

    def get_mini_batch(self, batch_idx):
        cur_batch = self.inputs['train'][batch_idx].asnumpyarray()
        #print cur_batch
        batch_size = cur_batch.shape[1]
        # cur_batch = cur_batch[~np.isnan(cur_batch)]
        # print cur_batch.shape
        input_batch = np.zeros((FEATURE_LENGTH, batch_size))
        for col in range(batch_size):
            bin_idx = cur_batch[:, col]
            input_batch[bin_idx[~np.isnan(bin_idx)].astype(int), :] = 1
            # for row in range(cur_batch.shape[0]):
            #     if ~np.isnan(cur_batch[row, col].asnumpyarray()):
            #         input_batch[row, col] = 1;
        return self.backend.array(input_batch), self.targets['train'][batch_idx]

    def get_batch(self, data, batch):
        """
        Extract and return a single batch from the data specified.

        Arguments:
            data (list): List of device loaded batches of data
            batch (int): 0-based index specifying the batch number to get

        Returns:
            neon.backends.Tensor: Single batch of data

        See Also:
            transpose_batches
        """
        return data[batch]

class FlyPredict(Dataset):
    """
    Sets up the fly v fly dataset.

    Attributes:
        backend (neon.backends.Backend): backend used for this data
        inputs (dict): structure housing the loaded train/test/validation
                       input data
        targets (dict): structure housing the loaded train/test/validation
                        target data

    Kwargs:
        repo_path (str, optional): where to locally host this dataset on disk
    """

    def __init__(self, **kwargs):
        self.macro_batched = False
        self.dist_flag = False
        self.num_test_sample = 10000
        self.__dict__.update(kwargs)
        if self.dist_flag:
            raise NotImplementedError("Dist not yet implemented for Chess")
        # if not hasattr(self, 'save_dir'):
        #     self.save_dir = os.path.join(self.repo_path,
        #                                  self.__class__.__name__)

    def load(self):
        if self.inputs['train'] is not None:
            return

        flydata = load_data()
        self.inputs['train'] = np.vstack([flydata[i][0] for i in train_nos])
        self.targets['train'] = np.vstack([flydata[i][1] for i in train_nos])
        #print "Training size: ", self.inputs['train'].shape
        #self.inputs['validation'] = np.vstack([flydata[i][0] for i in validation_nos])
        #self.targets['validation'] = np.vstack([flydata[i][1] for i in validation_nos])  
        self.inputs['test'] = np.vstack([flydata[i][0] for i in test_nos])
        self.targets['test'] = np.vstack([flydata[i][1] for i in test_nos])
        print "Test Size: ", self.inputs['test'].shape
        self.test = np.vstack([flydata[i][1][:15000,:] for i in test_nos])
        self.format()

    def get_mini_batch(self, batch_idx):
        cur_batch = self.inputs['train'][batch_idx].asnumpyarray()
        #print cur_batch
        batch_size = cur_batch.shape[1]
        # cur_batch = cur_batch[~np.isnan(cur_batch)]
        input_batch = np.zeros((FEATURE_LENGTH, batch_size))
        for col in range(batch_size):
            bin_idx = cur_batch[:, col]
            input_batch[bin_idx[~np.isnan(bin_idx)].astype(int), :] = 1
            # for row in range(cur_batch.shape[0]):
            #     if ~np.isnan(cur_batch[row, col].asnumpyarray()):
            #         input_batch[row, col] = 1;
        return self.backend.array(input_batch), self.targets['train'][batch_idx]

    def get_batch(self, data, batch):
        """
        Extract and return a single batch from the data specified.

        Arguments:
            data (list): List of device loaded batches of data
            batch (int): 0-based index specifying the batch number to get

        Returns:
            neon.backends.Tensor: Single batch of data

        See Also:
            transpose_batches
        """
        return data[batch]

if __name__ == '__main__':
    load_data()
