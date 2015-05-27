import os
import logging
import math
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import pylab as plt
import sys
import matplotlib.cm as cm

logging.basicConfig(level=20,
                    format='%(asctime)-15s %(levelname)s:%(module)s - %(message)s')
logger = logging.getLogger('thread example')

# neon specific imports
from neon.backends.cpu import CPU
try:
    from neon.backends.cc2 import GPU
    be = GPU(rng_seed=0, seterr_handling={'all': 'warn'},datapar=False, modelpar=False,
      actual_batch_size=30)
except ImportError:
    be = CPU(rng_seed=0, seterr_handling={'all': 'warn'},datapar=False, modelpar=False,
      actual_batch_size=30)

from neon.backends.par import NoPar
from neon.models.mlp import MLP
from flyvflymulticlass import FlyPredict
from neon.util.persist import deserialize

NUM_CLASSES = 6

def compute_f1(precision, recall):
    f1 = 2*precision*recall / (precision + recalll)
    index = np.argmax(f1)
    return index, f1[index]    

def prc_curve(targets_ts, scores_ts, targets_tr, scores_tr, model_no):
    plt.clf()
    print targets_ts.shape
    print scores_ts.shape
    colors = ['r', 'g', 'b', 'y', 'k', 'm']
    classes = ['lunge', 'wing_threat', 'charge', 'hold', 'tussle', 'other']
    for i in range(NUM_CLASSES):
        precision_ts, recall_ts, thresholds = precision_recall_curve(targets_ts[:,i], scores_ts[:,i], pos_label=1)
        precision_tr, recall_tr, thresholds = precision_recall_curve(targets_tr[:,i], scores_tr[:,i], pos_label=1)
        area_ts = auc(recall_ts, precision_ts)
        area_tr = auc(recall_tr, precision_tr)
        f1_test = compute_f1(precision_ts, recall_ts)
        f2_test = compute_f1(precision_tr, recall_tr)
        plt.plot(recall_ts, precision_ts, '--',label="%s test AUC: %0.4f" %(classes[i], area_ts), 
            color=colors[i])
        plt.plot(recall_tr, precision_tr, label="%s train AUC: %0.4f" %(classes[i],area_tr),
            color=colors[i])
    plt.title('Precision Recall of MC Model ' + model_no)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc="lower left", prop={'size':8})
    plt.grid(b=True, which='major')
    figure = plt.gcf()
    figure.set_size_inches(8, 6)
    plt.savefig('PRC_mc_model' + model_no +'.png')

def test():

    model.print_layers()
    dataset = FlyPredict(backend=be)

    # par related init
    be.actual_batch_size = model.batch_size
    be.mpi_size = 1
    be.mpi_rank = 0
    be.par = NoPar()
    be.par.backend = be

    # for set_name in ['test', 'train']:
    model.data_layer.init_dataset(dataset)
    model.data_layer.use_set('train', predict=True)
    dataset.use_set = 'train'
    scores, targets = model.predict_fullset(dataset, "train")
    scores_tr = np.transpose(scores.asnumpyarray())
    targets_tr = np.transpose(targets.asnumpyarray())

    model.data_layer.use_set('test', predict=True)
    dataset.use_set = 'test'
    scores, targets = model.predict_fullset(dataset, "test")
    scores_ts = np.transpose(scores.asnumpyarray())
    targets_ts = np.transpose(targets.asnumpyarray())
    model_no = sys.argv[1].split(".")[0][-2:]
    prc_curve(targets_ts, scores_ts, targets_tr, scores_tr, model_no)

def visualize():
    weights = model.layers[-2].weights.asnumpyarray()
    plt.subplot(1, 2, 1)
    plt.imshow(np.transpose(np.sort(abs(model.layers[-2].weights.asnumpyarray()))), cmap = cm.Greys_r)
    plt.subplot(1, 2, 2)
    plt.imshow(np.sort(np.transpose(abs(model.layers[-3].weights.asnumpyarray()))), cmap = cm.Greys_r)
    plt.show()

    weights_sort = weights.argsort()
    max_weights = weights[0, weights_sort[0, -5:]]
    min_weights = weights[0, weights_sort[0, 0:5]]
    print min_weights
if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        model = deserialize(f)
    #visualize()
    test()
