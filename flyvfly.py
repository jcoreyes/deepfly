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

#MOVIE_DIR = "/home/coreyesj/flyvflydata/Aggression/Aggression"
MOVIE_DIR = expanduser("~") + "/flyvflydata/Aggression/Aggression"
# Feature constants
NUM_FRAMES = 3
FEATURE_LENGTH = 1 * 36 * NUM_FRAMES

movie_nos = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # Not zero index
train_nos = range(1,6) # Zero indexed
test_nos = range(5,11)
neg_frac = 0.6
pos_frac = 20.0
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

def transform(trk_data, labels, filter_flag, fly_no=None, window_length=3, stride=1):
    """Get sliding window of frames from tracking data"""

    num_frames, num_features = trk_data.shape[1:3]
    action_no = 0
    X = np.zeros((num_frames-window_length+1, FEATURE_LENGTH))
    Y = np.zeros((X.shape[0], 1))
    # Get window of tracking data over time
    for i in xrange(0, num_frames - window_length):
        window = trk_data[fly_no, i:(i+window_length), :]
        X[i, :] = np.reshape(window, (1, window.size))
    action_labels = labels[fly_no, action_no][:, 0:2]
    # Action labels format: num_frames x 3: [start_frame, end_frame, 0/1 for if fly switched]
    for i in xrange(0, len(action_labels)):
        start, stop = action_labels[i, :]
        Y[start:stop, 0] = 1
    if filter_flag:
        X, Y = filter_data(X, Y)
    return X, Y

def filter_data(X, Y):
    """Filter out percentage of data with no actions"""
    idx1 = np.where(Y == 0)[0]
    idx2 = np.where(Y == 1)[0]
    num_neg = int(neg_frac * idx1.shape[0])
    num_pos = int(pos_frac * idx2.shape[0]) 
    num_points = num_neg + num_pos
    print("Filtered out no action from %d to %d" %(len(idx1), num_neg))
    print("Percent with action is now %f instead of %f" %(num_pos / float(num_points),
        float(len(idx2)) / (len(idx2) + len(idx1))))
    idx1 = idx1[0:num_neg]
    newX = np.zeros((num_points, X.shape[1]))
    newX[:len(idx1), :] = X[idx1, :]
    newX[len(idx1):,:] = np.tile(X[idx2, :], (int(pos_frac), 1))
    newY = np.zeros((num_points, 1))
    newY[:len(idx1), 0] = Y[idx1,0]
    newY[len(idx1):,0] = np.tile(Y[idx2, 0], (1, int(pos_frac)))
    return newX, newY

def load_data(input_movie_nos, filter_flag=None):
    data = []
    for movie_no in input_movie_nos:
        trk_data = read_tracking_data(movie_no)
        trk_data[np.isnan(trk_data)] = 0
        labels = read_labels(movie_no)[1]
        data.append(transform(trk_data, labels, filter_flag, fly_no = 0))
        data.append(transform(trk_data, labels, filter_flag, fly_no = 1))
    return data


class Fly(Dataset):
    """
    Sets up the fly v fly dataset.
    """

    def __init__(self, **kwargs):
        self.macro_batched = False
        self.dist_flag = False
        self.num_test_sample = 10000
        self.__dict__.update(kwargs)
        if self.dist_flag:
            raise NotImplementedError("Dist not yet implemented for Fly")
        # if not hasattr(self, 'save_dir'):
        #     self.save_dir = os.path.join(self.repo_path,
        #                                  self.__class__.__name__)

    def load(self):
        if self.inputs['train'] is not None:
            return
        train_x, train_y = zip(*load_data(train_nos,filter_flag=True))
        self.inputs['train'] = np.vstack(train_x)
        self.targets['train'] = np.vstack(train_y)
        print "Training size: ", self.inputs['train'].shape
        
        #test_x, test_y = zip(*load_data(test_nos))
        #self.inputs['test'] = np.vstack(test_x)
        #self.targets['test'] = np.vstack(test_y)
        #print "Test Size: ", self.inputs['test'].shape
        self.format()

    def get_mini_batch(self, batch_idx):
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
        global pos_frac
        pos_frac = 1.0
        train_x, train_y = zip(*load_data(train_nos, filter_flag=True))
        self.inputs['train'] = np.vstack(train_x)
        self.targets['train'] = np.vstack(train_y)
        print "Training size: ", self.inputs['train'].shape
        
        test_x, test_y = zip(*load_data(test_nos, filter_flag=False))
        self.inputs['test'] = np.vstack(test_x)
        self.targets['test'] = np.vstack(test_y)
        print "Test Size: ", self.inputs['test'].shape
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
