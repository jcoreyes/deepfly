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
from flyvfly import FlyPredict
from neon.util.persist import deserialize

def prc_curve(targets_ts, scores_ts, targets_tr, scores_tr, model_no):
    precision_ts, recall_ts, thresholds = precision_recall_curve(targets_ts, scores_ts, pos_label=1)
    precision_tr, recall_tr, thresholds = precision_recall_curve(targets_tr, scores_tr, pos_label=1)
    area_ts = auc(recall_ts, precision_ts)
    area_tr = auc(recall_tr, precision_tr)
    print precision_ts[len(precision_ts)-5:]
    print precision_tr[len(precision_tr)-5:]
    plt.clf()
    plt.plot(recall_ts, precision_ts, label="Test AUC: %f" %area_ts)
    plt.plot(recall_tr, precision_tr, label="Train AUC: %f" %area_tr)
    plt.title('Precision Recall of Model ' + model_no)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc="lower left")
    plt.grid(b=True, which='major')
    plt.savefig('PRC_model' + model_no +'.png')

def test():
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
    np.savetxt("model23weights2.txt", weights)
    np.savetxt("model23weights1.txt", model.layers[-3].weights.asnumpyarray())
    #plt.subplot(1, 2, 1)
    #plt.imshow(np.transpose(abs(model.layers[-2].weights.asnumpyarray())), cmap = cm.Greys_r)
    #plt.subplot(1, 2, 2)
    plt.imshow(abs(model.layers[-3].weights.asnumpyarray()), cmap = cm.Greys_r)
    plt.xlabel("Input Index")
    plt.ylabel("NN Unit")
    plt.savefig('weights.png', bbox_inches='tight');
    #plt.show()
    NUM_W = 5
    weights_sort = weights.argsort()
    max_weights = weights[0, weights_sort[0, -NUM_W:]]
    min_weights = weights[0, weights_sort[0, 0:NUM_W]]
    max_weights_index = weights_sort[0, -NUM_W:].tolist()
    max_weights_index += weights_sort[0, 0:NUM_W].tolist()
    print "Max and min weights"
    print max_weights
    print min_weights
    print "Max and min weights index", max_weights_index
    assert len(max_weights_index) == NUM_W*2
    print "Weight shape", weights.shape
    model.data_layer.init_dataset(dataset)
    model.data_layer.use_set('train', predict=True)
    batch = 0
    max_input = [[] for x in range(NUM_W*2)]
    print "Prec act shape", model.layers[-3].pre_act.asnumpyarray().shape
    #print model.layers[-3].pre_act.shape
    for batch_preds, batch_refs in model.predict_generator(dataset,
                                                          'train'):
        start = batch * model.batch_size
        end = start + model.batch_size
        output = model.get_classifier_output()
        # Prev output shape is num_neurons x batch_size so each row is 1 neuron

        prev = model.layers[-3].pre_act.asnumpyarray()[max_weights_index, :]
        curr_max_input = (np.argmax(prev, axis=1) + batch*30, np.amax(prev, axis=1))
        #print curr_max_input[0].shape
        for w in range(NUM_W*2):
            max_input[w].append((curr_max_input[0][w], curr_max_input[1][w]))
        batch += 1
    print batch
    points_per_vid = 54000*2
    with open('max_input2.txt', 'w') as f:       
        for w in range(NUM_W*2):
            max_input[w].sort(key=lambda x:x[1], reverse=True)
            frame_idx = [x[0] for x in max_input[w][0:5]]
            for frame in frame_idx:
                f.write("m%d f%d " %(int(frame/float(points_per_vid))+1, int((frame %points_per_vid)/2.0)))
            f.write("\n")

def find_frame_act():
    # Last layer is cost layer, so second to last is last weight layer
    #weights = model.layers[-2].weights.asnumpyarray()

    max_weights_index = [77, 90, 66, 46]
    NUM_W = len(max_weights_index)
    model.data_layer.init_dataset(dataset)
    model.data_layer.use_set('train', predict=True)
    batch = 0
    max_input = [[] for x in range(NUM_W)]
    print "Prec act shape", model.layers[-3].pre_act.asnumpyarray().shape
    #print model.layers[-3].pre_act.shape
    for batch_preds, batch_refs in model.predict_generator(dataset,
                                                          'train'):
        start = batch * model.batch_size
        end = start + model.batch_size
        output = model.get_classifier_output()
        # Prev output shape is num_neurons x batch_size so each row is 1 neuron

        prev = model.layers[-3].pre_act.asnumpyarray()[max_weights_index, :]
        curr_max_input = (np.argmax(prev, axis=1) + batch*30, np.amax(prev, axis=1))
        #print curr_max_input[0].shape
        for w in range(NUM_W):
            max_input[w].append((curr_max_input[0][w], curr_max_input[1][w]))
        batch += 1
    print batch
    points_per_vid = 54000*2
    with open('max_input2.txt', 'w') as f:       
        for w in range(NUM_W):
            max_input[w].sort(key=lambda x:x[1], reverse=True)
            frame_idx = [x[0] for x in max_input[w][0:5]]
            for frame in frame_idx:
                f.write("m%d f%d " %(int(frame/float(points_per_vid))+1, int((frame %points_per_vid)/2.0)))
            f.write("\n")

if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        model = deserialize(f)


    model.print_layers()
    # par related init
    be.actual_batch_size = model.batch_size
    be.mpi_size = 1
    be.mpi_rank = 0
    be.par = NoPar()
    be.par.backend = be

    dataset = FlyPredict(backend=be)
    visualize()
    #test()
