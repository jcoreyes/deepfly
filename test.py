import os
import logging
import math
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import pylab as plt
import sys

logging.basicConfig(level=20,
                    format='%(asctime)-15s %(levelname)s:%(module)s - %(message)s')
logger = logging.getLogger('thread example')

# neon specific imports
from neon.backends.cpu import CPU
from neon.backends.cc2 import GPU
from neon.backends.par import NoPar
be = GPU(rng_seed=0, seterr_handling={'all': 'warn'},datapar=False, modelpar=False,
  actual_batch_size=30)
from neon.models.mlp import MLP
from flyvfly import FlyPredict
from neon.util.persist import deserialize

def prc_curve(targets_ts, scores_ts, targets_tr, scores_tr, model_no):
    precision_ts, recall_ts, thresholds = precision_recall_curve(targets_ts, scores_ts, pos_label=1)
    precision_tr, recall_tr, thresholds = precision_recall_curve(targets_tr, scores_tr, pos_label=1)
    #area = auc(recall, precision)
    print precision_ts[len(precision_ts)-5:]
    print precision_tr[len(precision_tr)-5:]
    plt.clf()
    plt.plot(recall_ts, precision_ts, label="Test")
    plt.plot(recall_tr, precision_tr, label="Train")
    plt.title('Precision Recall of Model ' + model_no)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc="lower left")
    plt.savefig('PRC_model' + model_no +'.png')

def test():
    with open(sys.argv[1], 'r') as f:
        model = deserialize(f)
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
    
if __name__ == '__main__':
    test()
