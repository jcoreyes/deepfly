import os
import logging
import math
import sys
import numpy as np
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
from neon.layers import FCLayer, DataLayer, CostLayer
from neon.models.mlp import MLP
from neon.params.val_init import NodeNormalizedValGen
from neon.transforms.rectified import RectLin
from neon.transforms.linear import Linear
from neon.transforms.logistic import Logistic
from neon.transforms.cross_entropy import CrossEntropy
from neon.optimizers import GradientDescentMomentum
from flyvflymulticlass import Fly
from neon.util.persist import serialize
from neon.util.persist import deserialize

MINIBATCH_SIZE = 30
WINDOW_LENGTH = 3
USE_BOTH = False
FEATURE_LENGTH = (USE_BOTH+1) * 36 * WINDOW_LENGTH
NUM_CLASSES = 6

def get_parameters(n_in=None, n_hidden_units = 100,  n_hidden_layers=None):
    print 'initializing layers'
    if type(n_hidden_units) != list:
        n_hidden_units = [n_hidden_units] * n_hidden_layers
    else:
        n_hidden_layers = len(n_hidden_units)

    wt_init0 = NodeNormalizedValGen(backend=be, scale=1.0, bias_init=0.1)
    wt_init1 = NodeNormalizedValGen(backend=be, scale=1.0, bias_init=0.1)
    # in original learning_rate was exponentially decaying with 0.03 but with
    # 3x updates/mb
    gdmwd = {'type': 'gradient_descent_momentum',
             'lr_params': {'learning_rate': 0.005, 'backend': be,
                            'weight_decay': 1.0,
                           'momentum_params': {'type': 'constant', 'coef': 0.9}}}
    dataLayer = DataLayer(name='d0', nout=n_in)
    layers = []
    layers.append(dataLayer)
    for l in xrange(n_hidden_layers):
        if l < n_hidden_layers - 1:
            layers.append(FCLayer(name='h' + str(l),
                                  nout=n_hidden_units[l],
                                  lrule_init=gdmwd,
                                  weight_init=wt_init0,
                                  activation=Logistic()))
        else:
            layers.append(FCLayer(name='h' + str(l),
                                  nout=n_hidden_units[l],
                                  lrule_init=gdmwd,
                                  weight_init=wt_init1,
                                  activation=Logistic()))

    # add CostLayer
    layers.append(CostLayer(name='cost', ref_layer=dataLayer,
                            cost=CrossEntropy()))
    return layers

def get_validation(model, dataset):
    model.data_layer.use_set('validation', predict=True)
    dataset.use_set = 'validation'
    model.predict_fullset(dataset, "validation")
    estim, targets = map(lambda x: x.asnumpyarray(), model.predict_fullset(dataset, "validation"))
    estim[estim>0.5] = 1
    estim[estim<=0.5] = 0
    val_err = np.sum(estim != targets) / float(estim.shape[1])
    return val_err


def train():

    save_file = sys.argv[1]
    if len(sys.argv) > 2:
        model = deserialize(sys.argv[2])
    else:
        layers = get_parameters(n_in=FEATURE_LENGTH, n_hidden_units=[200,100, NUM_CLASSES])
        # define model
        model = MLP(num_epochs=1, batch_size=MINIBATCH_SIZE,
                     layers=layers, epochs_complete=0)
        model.link()
        #be.configure(model, datapar=False, modelpar=False)
        model.initialize(be)
        model.data_layer = model.layers[0]
        model.cost_layer = model.layers[-1]

    dataset = Fly(backend=be,
                    repo_path=os.path.expanduser('~/flyvfly/'))
    
    # par related init
    be.actual_batch_size = model.batch_size
    be.mpi_size = 1
    be.mpi_rank = 0
    be.par = NoPar()
    be.par.backend = be

    max_macro_epochs = 1000
    min_err = sys.maxint
    for i in range(max_macro_epochs):
        model.epochs_complete = 0
        dataset.use_set = "train"
        model.fit(dataset)
        #scores, targets = model.predict_fullset(dataset, "validation")
        val_err = get_validation(model, dataset)
        logger.info('epoch: %d,  valid error: %0.6f', i, val_err)
        if val_err < min_err:
            serialize(model, save_file)
            min_err = val_err

if __name__ == '__main__':
    train()
