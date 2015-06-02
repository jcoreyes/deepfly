import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from flyvfly import read_tracking_data
from matplotlib.pylab import *



def plot_matrix(w1, title, savefile, xlabels=[], ylabels=[]):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(w1)
    start, end = ax.get_xlim()
    #ax.xaxis.set_ticks(np.arange(start, end, 3))
    #ax.yaxis.set_ticks([])
    ax.grid(False)
    #ax.xaxis.set_ticklabels([])
    #ax.xaxis.set_ticklabels(trk_names)
    #fig.colorbar(cax)
    #plt.ylabel("Unit Index")
    #print xlabels
    plt.xticks(range(w1.shape[1]), xlabels, rotation='horizontal')
    plt.yticks(range(w1.shape[0]), ylabels, rotation='horizontal')
    #plt.xlabel("Weights")
    plt.title(title)
    #plt.savefig(savefile, bbox_inches='tight')
  
    show()

def viz_weights():
    # Plot values of weights
    #plot_matrix(w1, "Value of weights", "viz/weights_value.png")

    # Plot value of weights ordered P/V/A
    w1o1 = w1.copy()
    for start in [12, 48, 84]:
        w1o1[:, start:start+12] = w1[:, start:start+24:2]
        w1o1[:, start+1:start+13] = w1[:, start+1:start+25:2]
    #plot_matrix(w1o1, "Value of weights ordered P/V/A at window level", "viz/weights_value_ordered1.png")

    # Plot value of weights odred P/V/A at frame level
    index = []
    for i in range(36):
        index+= range(i, 108, 36)
    assert len(index) == 108
    #plot_matrix(w1o1[:, index], "Value of weights ordered P/V/A at frame level", "viz/weights_value_ordered2.png")


    # Filter out units with magnitude less than mean
    means = np.mean(np.abs(w1), axis=1)
    unit_index = means>1.1*np.mean(means)
    #plot_matrix(w1[means>1.2*np.mean(means), :], "Filtered weights", "viz/weights_value_filtered.png")
    w1fo1 = w1o1[unit_index, :]
    #plot_matrix(w1fo1, "Value of filtered weights ordered P/V/A at window level", "viz/weights_value_filtered_ordered1")
    w1fo1 = w1fo1[:, index]
    #plot_matrix(w1fo1, "Value of weights filtered ordered P/V/A at frame level", "viz/weights_value_filtered_ordered2")

    # Sort by w
    max_w = np.sum(np.abs(w1fo1), axis = 0).argsort()
    print max_w
    assert max_w.shape[0] == w1fo1.shape[1]
    #w1fo1 = w1fo1[:, indx]

    # Filter out features
    feature_means = np.zeros((100, 36))
    w1f2 = np.abs(w1fo1)
    feature_sums = (w1f2[:, 0:36] + w1f2[:, 36:72] + w1f2[:, 72:])/3
    feature_means = feature_sums.mean(axis=0)
    args = feature_means.argsort()
    for x in range(-1, -6, -1):
        print trk_names_lst[args[x]], feature_means[args[x]]
    #print [trk_names_lst[args[-x]] for x in range(5)]
    #plot_matrix(feature_sums, "Average magnitude of each feature weight over 3 frames", "viz/feature_means_filtered")

    pav_sums = np.zeros((100, 9))
    for i, start in enumerate(range(0, 108, 12)):
        pav_sums[:, i] = np.sum(np.abs(w1o1[:, start:start+12]), axis=1)
    pav_sums_i = [0, 3, 6, 1, 4, 7, 2, 5, 8]
    #plot_matrix(pav_sums[:, pav_sums_i], "Average magnitude of P/V/A", "viz/feature_means_ordered1")

    frame_sums = np.zeros((100, 3))
    for i, start in enumerate(range(0, 108, 36)):
        frame_sums[:, i] = np.sum(np.abs(w1o1[:, start:start+36]), axis=1)
    plot_matrix(frame_sums, "Average magnitude of frame weights", "viz/frame_means")

    #plot_matrix(w2, "Value of weights", "viz/weights2_value.png")
    #
    max_w = np.sum(np.abs(w1), axis = 1).argsort()
    w1 = w1[max_w, :]
    plt.imshow(w1, cmap = cm.Greys_r)
    #plt.show()
    max_w = np.sum(np.abs(w1), axis = 1)
    top = max_w.argsort()[-5:]
    #print max_w
    #print top

    #plt.imshow(np.reshape(w1[top[0], :], (3, 36)), cmap = cm.Greys_r)
    #plt.show()

def find_max_features(w1, w2):
    max_w_index = w2.argsort()[0, ::-1]
    min_max_index = range(20) + range(70, 99)
    max_w_index = max_w_index[min_max_index]
    max_w = w2[0, max_w_index]
    print max_w
    print max_w_index
    print w1.shape
    w1o1 = w1.copy()
    for start in [12, 48, 84]:
        w1o1[:, start:start+12] = w1[:, start:start+24:2]
        w1o1[:, start+1:start+13] = w1[:, start+1:start+25:2]
    #plot_matrix(w1o1, "Value of weights ordered P/V/A at window level", "viz/weights_value_ordered1.png")

    # Plot value of weights odred P/V/A at frame level

    w1 = w1[:, index]

    w1f = w1[max_w_index, :]



    filter_index_b = np.abs(w1f).mean(axis=0) > np.abs(w1f).mean()
    filter_index = np.arange(0, 108)[filter_index_b]
    plot_matrix(np.transpose(w1f[:, filter_index_b]), "", "", 
        ylabels=[trk_names[filter_index[i]][0] + " %d" % int(filter_index[i]%3+1) for i in range(len(filter_index))], 
        xlabels=max_w_index)
    
    corr_w = w1[max_w_index[1:3], :]
    filter_index = np.abs(corr_w).mean(axis=0) > np.abs(corr_w).mean()
    corr_diff = [corr_w[:, filter_index], np.abs(corr_w[0, filter_index] - corr_w[1, filter_index])]

    filter_index = np.arange(0, 108)[filter_index]
    ylabels = [trk_names[0, filter_index[i]][0] + " %d" % int(filter_index[i]%3+1) for i in range(len(filter_index))]

    plot_matrix(np.transpose(np.vstack(corr_diff)), "", "", 
        ylabels=ylabels, xlabels=["w%d" %max_w_index[1], "w%d" %max_w_index[2], "diff"])

if __name__ == '__main__':
    index = []
    for i in range(36):
        index+= range(i, 108, 36)
    assert len(index) == 108
    trk_data, trk_names = read_tracking_data(1)
    trk_names = np.hstack([trk_names, trk_names, trk_names])
    trk_names = trk_names[0, index]
    #print trk_names.shape

    # trk_names_lst = [str(trk_names[0,x]) for x in range(trk_names.shape[1])]
    # for i,v in enumerate(trk_names_lst):
    #     print i, v.encode( 'utf-8' )

    w1 = np.loadtxt("model23weights1.txt")
    w2 = np.loadtxt("model23weights2.txt")
    w2 = w2.reshape((1, w2.shape[0]))

    find_max_features(w1, w2)
