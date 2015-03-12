from solver import *
import lasagne
import theano
import theano.tensor as T
import sys
sys.path.insert(0,'../')
from caffe2theano import models
import numpy as np
import caffe2theano
import dataset

lmodel = caffe2theano.conversion.convert('/root/caffe/models/bvlc_reference_caffenet/deploy.prototxt','/root/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
ds = dataset.ImageDirectoryDataset('../../proj/Images', 1000, 250)


inp = lasagne.layers.InputLayer(ds.X_batch_shape)
reshape_layer = lasagne.layers.ReshapeLayer(lmodel.input_layer, ([0]*[1], [2], [3], [4]))

# ======= [ NET SURGERY. MUST TEST. ] ====== #
all_lmodel_layers = lmodel.layers
without_input = all_model_layers[1:]
new_layers = [inp, reshape_layer] + without_input

without_input[0].input_layer = reshape_layer
without_input[0].input_shape = reshape_layer.get_output_shape()

lmodel.__init__(lmodel.last_layer)
