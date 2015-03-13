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

class GatedSingleInputLayer(lasagne.layers.Layer):
    '''
    A layer that takes in multiple inputs *of the same dimensionality* and computes gates to combine them
    '''
    def __init__(self, incoming, W=init.Uniform(), b = init.Constant(0.), nonlinearity=lasagne.nonlinearities.sigmoid, prob_func=lasagne.nonlinearities.softmax, **kwargs):
        super(GatedMultipleInputsLayer,self).__init__(incomings,**kwargs)
        num_out = self.input_shape[2]
        # make gates
        self.W = self.create_param(W, (num_out,num_out))
        self.b = self.create_param(b, (num_out,))

        self.nonlinearity = nonlinearity
        self.prob_func = prob_func


    def get_output_for(self, input, *args, **kwargs):
        # compute gates
        gs = self.nonlinearity(T.tensordot(input, self.W) + self.b.dimshuffle('x','x',0))
        # gs is 10 x 36 x 4096
        # now turn to probability 

        # scan op to softmax each one
        def step(tens):
            out = self.prob_func(tens.transpose()).transpose()

        softmaxd = theano.scan(fn=step, sequences=gs) 
        # now softmaxd should be 10 x 36 x 4096, but the 4096 across 36 should sum to a probability distribution
        # now elementwise multiplication
        outs = softmaxd * input
        # now two options: 
            # 1. elementwise mult, just average and then go to softmax layer
            # 2. or, we could have calculated importance scores for each, and sent it to softmax layer and then elementwise mult with the importance scores. to try.
        return T.mean(outs, axis=1) # should be 10 x 4096

    def get_params(self):
        return self.W + self.b

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])




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

# now we continue to the gates
gates = GatedSingleInputLayer(last_reshape)
# dense and softmax (maybe one more dense???)
dense = lasagne.layers.DenseLayer(gates, 67, nonlinearity=lasagne.nonlinearities.softmax)

lmodel.__init__(dense)
