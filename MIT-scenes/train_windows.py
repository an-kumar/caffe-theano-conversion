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
ds = dataset.ImageDirectoryDataset('../../proj/Images', 250, 50)

batchsh = list(ds.X_batch_shape)
batchsh[0] = None
batchsh = tuple(batchsh)
inp = lasagne.layers.InputLayer(batchsh)
reshape_layer = lasagne.layers.ReshapeLayer(inp, (-1, 3, 227, 227))

# ======= [ NET SURGERY. MUST TEST. ] ====== #
all_lmodel_layers = lmodel.all_layers
without_input = all_lmodel_layers[:-1]
# 7: - before relu (first fc)
# 6: - after relu (first fc)
# 4: - before relu (second fc)
# 3: - after relu(second fc)
without_output = without_input[7:] # cut off softmax, last fully connected, 
last_reshape = lasagne.layers.ReshapeLayer(without_output[0], (-1, 36, 4096)) # watch these hardcoded numbers, 36 and 4096.
new_layers = [last_reshape] + without_output + [reshape_layer, inp]

without_output[-1].input_layer = reshape_layer
for layer in new_layers[:-1][::-1]: # must reverse so that the shapes are correct as we go up.
    layer.input_shape = layer.input_layer.get_output_shape()

lmodel.__init__(last_reshape)
