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
NUM_BINS = 5
movie_nos = [1, 2, 3, 4, 5, 6]

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
    disc_trk_data = np.zeros((2, num_frames*num_bins, num_features))
    binary_offsets = np.arange(0, num_bins*num_frames, num_bins)

    for j in xrange(num_features):
        # bin_nos will range from 0 to num_bins and be 0 or num_bins if outside range
        for fly_no in range(2):
            bin_nos = np.searchsorted(feature_bins[j,:], trk_data[fly_no,:, j])
            binary_index = bin_nos + binary_offsets
            binary_index = binary_index[~np.isnan(binary_index)]
            disc_trk_data[fly_no, binary_index[:-1], j] = 1

    # Keep nan's in trk_data as nans in disc
    #disc_trk_data[np.isnan(trk_data)] = np.nan

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

    X = np.zeros((num_frames-window_length+1, 2*num_features*window_length*NUM_BINS))
    Y = np.zeros((X.shape[0], 1))

    # Get window of tracking data over time
    for i in xrange(0, num_frames - window_length):
        window = trk_data[:, i*NUM_BINS:(i+window_length)*NUM_BINS, :]
        if window.size != window_length * NUM_BINS:
            continue
        X[i, :] = np.reshape(window, (1, window.size))

    action_labels = labels[fly_no, action_no][:, 0:2]
    for i in xrange(0, len(action_labels)):
        start, stop = action_labels[i, :]
        Y[start:stop, 0] = 1

    return X, Y

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
        self.__dict__.update(kwargs)

    def load(self):
        if self.inputs['train'] is not None:
            return
        train_nos = [0, 1, 2, 3]
        validation_nos = [4]
        test_nos = [5]

        flydata = load_data()
        self.inputs['train'] = np.vstack([flydata[i][0] for i in train_nos])
        self.targets['train'] = np.vstack([flydata[i][1] for i in train_nos])
        print "data shape"
        print self.inputs['train'].shape
        print self.inputs['train'].size
        print self.targets['train'].shape
        #self.inputs['validation'] = np.vstack([flydata[i][0] for i in validation_nos])
        #self.targets['validation'] = np.vstack([flydata[i][1] for i in validation_nos])  

        self.inputs['test'] = np.vstack([flydata[i][0] for i in test_nos])
        self.targets['test'] = np.vstack([flydata[i][1] for i in test_nos])

        self.format()

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
