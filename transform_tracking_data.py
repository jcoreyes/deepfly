"""
Parse pose tracking data downloaded from 
http://www.vision.caltech.edu/Video_Datasets/Fly-vs-Fly/download.html
and transforms frames of tracking data into vectors 
"""
import scipy.io
import numpy as np
import sys
MOVIE_DIR = "/home/coreyesj/flyvflydata/Aggression/Aggression"
movie_nos = [1, 2, 3, 4, 5, 6]

def read_tracking_data(movie_no):
    """ Read tracking data from matlab structs"""

    matpath = "%s/movie%d/movie%d_track.mat" %(MOVIE_DIR, movie_no, movie_no)
    matfile = scipy.io.loadmat(matpath, struct_as_record=True)
    trk_flagframes, trk_names, trk_data = matfile['trk'][0,0]
    return (trk_flagframes, trk_names, trk_data)

def discretize(trk_data, num_bins=5):
    """ Discretize tracking data into binary bins"""

    num_frames, num_features = trk_data.shape[1:3]

    # Create bins for each feature based on max and min ranges
    feature_bins = np.zeros((num_features, num_bins))
    for i in xrange(num_features):
        feature_bins[i, :] = np.linspace(min_range[i], max_range[i], num=num_bins)

    # Get bin number that each feature falls in
    disc_trk_data = np.zeros(trk_data.shape)
    
    for j in xrange(num_features):
        # bin_nos will range from 0 to num_bins and be 0 or num_bins if outside range
        for fly_no in range(2):
            bin_nos = np.digitize(trk_data[fly_no,:, j], bins=feature_bins[j, :])
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

def transform(trk_data, window_length=3, stride=1):
    """Get sliding window of frames from tracking data"""

    num_frames = trk_data.shape[1]
    # Get window of tracking data over time
    for i in xrange(0, num_frames - window_length, stride):
        window = trk_data[:, i:i+window_size, :]
        np.reshape(window, (1, window.size))

if __name__ == '__main__': 
    min_range, max_range = find_ranges()
    trk_flagframes, trk_names, trk_data = read_tracking_data(1)
    trk_data = discretize(trk_data, num_bins=5)
    transform(trk_data)