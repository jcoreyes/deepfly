"""
Caltech FlyvFly Dataset for preparing to be used by Neon.
More info at: http://www.vision.caltech.edu/Video_Datasets/Fly-vs-Fly/download.html
"""

import logging
import numpy as np, h5py 
import scipy.io
import random
from neon.datasets.dataset import Dataset
import cProfile
import matplotlib.pyplot as plt
from os.path import expanduser
import numpy as np
from sklearn import preprocessing
import gc

MOVIE_DIR = expanduser("~") + "/flyvflydata/Aggression/Aggression"
# Feature constants
WINDOW_LENGTH = 5 # Number of contig. frames to consider for 1 data point
USE_BOTH = False # Whether to use both fly's data for the same data point
FEATURE_LENGTH = (USE_BOTH+1) * 36 * WINDOW_LENGTH

movie_nos = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
train_nos = range(1, 6) # Zero indexed
validation_nos = [6]
test_nos = range(6,11)
action_nos = [0, 1, 2, 3, 4]
use_trk = False
logger = logging.getLogger(__name__)

def read_tracking_data(movie_no):
    """ Read tracking data from matlab structs
        Read trajectory data is use_trk is False"""
    if use_trk:
        matpath = "%s/movie%d/movie%d_track.mat" %(MOVIE_DIR, movie_no, movie_no)
        matfile = scipy.io.loadmat(matpath, struct_as_record=True)
        trk_flagframes, trk_names, trk_data = matfile['trk'][0,0]
    else:
        matpath = "%s/movie%d/movie%d_feat.mat" %(MOVIE_DIR, movie_no, movie_no)
        matfile = scipy.io.loadmat(matpath, struct_as_record=True)
        trk_names, trk_data = matfile['feat'][0,0]
    return trk_data

def read_labels(movie_no):
    """ Read action labels from matlab structs"""

    matpath = "%s/movie%d/movie%d_actions.mat" %(MOVIE_DIR, movie_no, movie_no)
    matfile = scipy.io.loadmat(matpath,squeeze_me=False)
    action_names, action_labels = matfile['behs'], matfile['bouts']
    return (action_names, action_labels)

def transform(trk_data, labels, filter_flag, fly_no=None, window_length=WINDOW_LENGTH, stride=1):
    """Get sliding window of frames from tracking data"""

    num_frames, num_features = trk_data.shape[1:3]
    X = np.zeros((num_frames-window_length+1, FEATURE_LENGTH))
    # Get window of tracking data over time
    for i in xrange(0, num_frames - window_length):
        if USE_BOTH:
            window = trk_data[:, i:(i+window_length), :]
            if fly_no == 0:
                X[i, :] = np.reshape(window, (1, window.size))
            else:
                X[i, :] = np.fliplr(np.reshape(window, (1, window.size)))
        else:    
            window = trk_data[fly_no, i:(i+window_length), :]
            X[i, :] = np.reshape(window, (1, window.size))
    half_window = int(WINDOW_LENGTH / 2.0)
    # Last col of Y is for no action
    Y = np.zeros((X.shape[0], len(action_nos) + 1))
    # Action labels format: num_frames x 3: [start_frame, end_frame, 0/1 for if fly switched]
    for index, action_no in enumerate(action_nos):
        action_labels = labels[fly_no, action_no][:, 0:2]
        for i in xrange(0, action_labels.shape[0]):
            start, stop = action_labels[i, :]
            # Use center frame in window as label so frames [0, 1, 2] would use label for
            # frame index 1
            if start - half_window >= 0:
                start -= half_window
                stop -= half_window
            Y[start:stop, index] = 1
    # Set no action
    Y[np.sum(Y, axis=1) == 0, -1] = 1

    if filter_flag:
        #X, Y = filter_data(X, Y)
        X, Y = filter_data(*replicationActions(X, Y))
        gc.collect()

    return X, Y

def filter_data(X, Y):
    """Filter out percentage of data with no actions"""
    idx1 = np.where(Y[:, -1] == 1)[0] #np.where(np.sum(Y, axis=1) == 0)[0]
    idx2 = np.where(Y[:, -1] == 0)[0] #np.where(np.sum(Y, axis=1) != 0)[0]
    assert idx1.shape[0] + idx2.shape[0] == X.shape[0]
    #print idx1.shape
    idx1 = idx1[:int(0.6*idx1.shape[0])]    
    #print idx1.shape
    index = np.hstack([idx1, idx2])
    X = X[index, :]
    Y = Y[index, :]
    return X, Y
    #return np.vstack([X[idx1, :], X[idx2, :]]), np.vstack([Y[idx1, :], Y[idx2, :]])

def replicationActions(X, Y):
    """ Replication each action according to ratios"""
    #action_ratios = [16, 2, 3000, 190, 38]
    action_ratios = [8, 1, 1350, 90, 19]
    #action_ratios = [1, 1, 1, 1, 1]
    for i in action_nos:
        X, Y = replicationAction(X, Y, i, action_ratios[i])
    return X, Y

def replicationAction(X, Y, action_no, ratio):
    """ Replicate a single action"""
    assert ratio == int(ratio)
    assert X.shape[0] == Y.shape[0]
    idx1 = Y[:, action_no] == 1
    if np.sum(idx1) == 0:
        return X, Y
    #X = np.vstack([X, np.tile(X[idx1, :], (ratio, 1))])
    #Y = np.vstack([Y, np.tile(Y[idx1, :], (ratio, 1))])
    return np.vstack([X, np.tile(X[idx1, :], (ratio, 1))]), np.vstack([Y, np.tile(Y[idx1, :], (ratio, 1))])

    
def print_ratios(Y):
    for i in range(5):
        num_pos = np.sum(Y[:,i]==1)
        print ("Action %d percentage %f" %(i, num_pos/float(Y.shape[0])))

    
def load_data(input_movie_nos, filter_flag=None):
    data = []
    for movie_no in input_movie_nos:
        trk_data = read_tracking_data(movie_no)
        trk_data[np.isnan(trk_data)] = 0
        standard(trk_data)
        labels = read_labels(movie_no)[1]
        data.append(transform(trk_data, labels, filter_flag, fly_no = 0))
        data.append(transform(trk_data, labels, filter_flag, fly_no = 1))
    return data

def standard(trk_data):
    """ Standardize the data to have zero mean and unit variance"""
    means = np.loadtxt("means.txt")
    for fly_no in range(2):
        scaler = preprocessing.StandardScaler()
        scaler.mean_ = means[fly_no]
        scaler.std_ = means[fly_no + 2]
        trk_data[fly_no, :, :] = scaler.transform(trk_data[fly_no, :, :])

class Fly(Dataset):
    """
    Sets up the fly v fly dataset.
    """

    def __init__(self, **kwargs):
        self.macro_batched = False
        self.dist_flag = False
        self.num_test_sample = 10000
        self.use_set = "train"
        self.__dict__.update(kwargs)
        if self.dist_flag:
            raise NotImplementedError("Dist not yet implemented for Fly")
        # if not hasattr(self, 'save_dir'):
        #     self.save_dir = os.path.join(self.repo_path,
        #                                  self.__class__.__name__)

    def load(self):
        if self.inputs['train'] is not None:
            return
        train_x, train_y = zip(*load_data(train_nos,filter_flag=False))
        self.inputs['train'] = np.vstack(train_x)
        self.targets['train'] = np.vstack(train_y)
        print "Training size: ", self.inputs['train'].shape
        print self.targets['train'].shape
        print_ratios(self.targets['train'])
        validation_x, validation_y = zip(*load_data(validation_nos,filter_flag=False))
        self.inputs['validation'] = np.vstack(validation_x)
        self.targets['validation'] = np.vstack(validation_y)
        print "Validation size: ", self.inputs['validation'].shape
        print self.targets['validation'].shape
        print_ratios(self.targets['validation'])
        #test_x, test_y = zip(*load_data(test_nos))
        #self.inputs['test'] = np.vstack(test_x)
        #self.targets['test'] = np.vstack(test_y)
        #print "Test Size: ", self.inputs['test'].shape
        self.format()

    def get_mini_batch(self, batch_idx):
        if self.use_set == 'validation':
            #print(len(self.inputs['validation'))
            #print batch_idx
            return self.inputs[self.use_set][batch_idx], self.targets[self.use_set][batch_idx]
        batch_idx = random.randint(0, len(self.inputs['train']) - 1)
        return self.inputs['train'][batch_idx], self.targets['train'][batch_idx]

    def get_batch(self, data, batch):
        """
        Extract and return a single batch from the data specified.
        """
        return data[batch]

class FlyPredict(Dataset):
    """
    Sets up the fly v fly dataset.
    """
    def __init__(self, **kwargs):
        self.macro_batched = False
        self.dist_flag = False
        self.num_test_sample = 10000
        self.use_set = "train"
        self.__dict__.update(kwargs)
        if self.dist_flag:
            raise NotImplementedError("Dist not yet implemented for Chess")
        # if not hasattr(self, 'save_dir'):
        #     self.save_dir = os.path.join(self.repo_path,
        #                                  self.__class__.__name__)

    def load(self):
        if self.inputs['train'] is not None:
            return

        train_x, train_y = zip(*load_data(train_nos, filter_flag=False))
        self.inputs['train'] = np.vstack(train_x)
        self.targets['train'] = np.vstack(train_y)
        print "Training size: ", self.inputs['train'].shape
        print_ratios(self.targets['train'])

        test_x, test_y = zip(*load_data(test_nos, filter_flag=False))
        self.inputs['test'] = np.vstack(test_x)
        self.targets['test'] = np.vstack(test_y)
        print "Test Size: ", self.inputs['test'].shape
        print_ratios(self.targets['test'])

        self.format()

    def get_mini_batch(self, batch_idx):
        return self.inputs[self.use_set][batch_idx], self.targets[self.use_set][batch_idx]


    def get_batch(self, data, batch):
        """
        Extract and return a single batch from the data specified.
        """
        return data[batch]

if __name__ == '__main__':
    load_data(train_nos, filter_flag=True)
