import theano
import numpy as np
import theano.tensor as T
try:
	import cc_layers
	cuda = True
except:
	print "probably no pylearn2; using normal"
	cuda = False
import layers
from parse_model_def import parse_model_def as parse
import caffe

def convert(prototxt, caffemodel):
	'''
	prototxt is a .prototxt file
	caffemodel is a .caffemodel file

	returns a function "forward", that's all. of course this code can be changed to do a lot of things.
	'''
	# parse the prototxt file
	input_dims, architecture = parse(prototxt)
	assert len(input_dims) == 4 #bc01
	net = caffe.Net(prototxt, caffemodel, caffe.TEST)

	# create input layer
	# this actually ends up being c01b shaped, but we pass in bc01
	if cuda == False:
		last_layer = inp_layer = layers.Input2DLayer(input_dims[0], input_dims[1], input_dims[2], input_dims[3])
	else:
		last_layer = inp_layer = cc_layers.CudaConvnetInput2DLayer(1, input_dims[1], input_dims[2], input_dims[3])

	# go thru layers and create the theano layer 
	all_layers = []
	swapped = False
	for layer in architecture:
		if layer['type'] == 'INNER_PRODUCT' and swapped==False and cuda==True:
			# need to add a reshaping layer
			reshape_layer = cc_layers.ShuffleC01BToBC01Layer(last_layer)
			all_layers.append(reshape_layer)
			last_layer = reshape_layer
			swapped = True
		this_layer = parse_layer(layer, last_layer)
		set_params(this_layer, net, layer)
		last_layer = this_layer
		all_layers.append(this_layer)

	X = T.tensor4('data', dtype=theano.config.floatX) # This will be the data we pass in; we could change this to an index into a batch for example, this is just for testing how this conversion script works
	givens = {inp_layer.input_var:X}
	forward = theano.function([X], [layer.output(dropout_active=False) for layer in all_layers],givens=givens)
	return forward, net, all_layers



def set_params(theano_layer, net, layer_params):
	'''
	theano_layer is a theano layer
	net is the caffe net
	layer_params is the layer params
	'''
	if len(theano_layer.params)== 0:
		return # no params to set
	else:
		if layer_params['type'] == 'CONVOLUTION':
			if cuda==True:
				set_cuda_conv_params(theano_layer, net, layer_params)
			else:
				set_conv_params(theano_layer, net, layer_params)
		elif layer_params['type'] == 'INNER_PRODUCT':
			set_ip_params(theano_layer, net, layer_params)
		else:
			print "not a valid layer to set params to (what happened??) %s" % layer_params['type']

def set_conv_params(theano_layer, net, layer_params):
	name = layer_params['name']
	W = net.params[name][0].data
	b = net.params[name][1].data
	# b needs to just be the last index
	b = b[0,0,0,:]
	theano_layer.W.set_value(W.astype(theano.config.floatX))
	theano_layer.b.set_value(b.astype(theano.config.floatX))

def set_cuda_conv_params(theano_layer, net, layer_params):
	name = layer_params['name']
	W = net.params[name][0].data
	b = net.params[name][1].data
	# b needs to just be the last index
	b = b[0,0,0,:]
	# W needs to be reshaped into n_features(from prev layer), size, size, n_filters
	W = W.transpose(1,2,3,0)
	theano_layer.W.set_value(W.astype(theano.config.floatX))
	theano_layer.b.set_value(b.astype(theano.config.floatX))

def set_ip_params(theano_layer, net, layer_params):
	name = layer_params['name']
	W = net.params[name][0].data
	b = net.params[name][1].data

	# W needs to just be the last 2, shuffled
	W = W[0,0,:,:].T
	# b needs to just be the last index
	b = b[0,0,0,:]
	theano_layer.W.set_value(W.astype(theano.config.floatX))
	theano_layer.b.set_value(b.astype(theano.config.floatX))


def parse_layer(layer, last_layer):
	'''
	returns the correct layer given the param dict
	'''
	if layer['type'] == 'CONVOLUTION':
		if cuda==True:
			return cuda_conv_layer_from_params(layer, last_layer)
		else:
			return conv_layer_from_params(layer, last_layer)
	elif layer['type'] == 'RELU':
		return relu_layer_from_params(layer, last_layer)
	elif layer['type'] == 'POOLING':
		if cuda==True:
			return cuda_pooling_layer_from_params(layer, last_layer)
		else:
			return pooling_layer_from_params(layer, last_layer)
	elif layer['type'] == 'INNER_PRODUCT':
		return ip_layer_from_params(layer, last_layer)
	elif layer['type'] == 'DROPOUT':
		return dropout_layer_from_params(layer, last_layer)
	elif layer['type'] == 'SOFTMAX':
		return softmax_layer_from_params(layer, last_layer)
	elif layer['type'] == 'LRN':
		return lrn_layer_from_params(layer, last_layer)
	else:
		print 'not a valid layer: %s' % layer['type']

def lrn_layer_from_params(layer, last_layer):
	lrn = layers.esponseNormalisationLayer(last_layer, int(layer['local_size']), 1, float(layer['alpha']), float(layer['beta']))
def cuda_conv_layer_from_params(layer, last_layer):
	''' CAN'T DO ANYTHING BUT (1,1) STRIDES RIGHT NOW! '''
	if layer['stride'] == 'DEFAULT':
		layer['stride'] = 1
	if int(layer['kernel_size']) - (int(layer['pad']) * 2 ) == 1:
		print "using same convolutions, this should be correct"
	else:
		print "this will be incorrect. the caffe net is not using same convolutions (i think). you should check what their doing, go into layers.py and fix this accordingly"
		raise Exception ("this will probably not work but try to comment this out if oyu want")
	conv = cc_layers.CudaConvnetConv2DLayer(last_layer, int(layer['num_output']), int(layer['kernel_size']), -1, -1, nonlinearity=layers.identity, pad=int(layer['pad']))
	return conv

def conv_layer_from_params(layer, last_layer):
	''' CAN'T DO ANYTHING BUT (1,1) STRIDES RIGHT NOW! '''
	if layer['stride'] == 'DEFAULT':
		layer['stride'] = 1
	if int(layer['kernel_size']) - (int(layer['pad']) * 2 ) == 1:
		print "using same convolutions, this should be correct"
	else:
		print "this will be incorrect. the caffe net is not using same convolutions (i think). you should check what their doing, go into layers.py and fix this accordingly"
		raise Exception ("this will probably not work but try to comment this out if oyu want")
	conv = layers.Conv2DLayer(last_layer, int(layer['num_output']), int(layer['kernel_size']), int(layer['kernel_size']), -1, -1, nonlinearity=layers.identity, border_mode='same')
	return conv

def relu_layer_from_params(layer, last_layer):
	return layers.ReluLayer(last_layer)

def pooling_layer_from_params(layer, last_layer):
	if layer['stride'] == 'DEFAULT':
		layer['stride'] = 1
	pool = layers.Pooling2DLayer(last_layer, (int(layer['kernel_size']), int(layer['kernel_size'])))
	return pool

def cuda_pooling_layer_from_params(layer, last_layer):
	if layer['stride'] == 'DEFAULT':
		layer['stride'] = 1
		print "this is a problem."
	pool = cc_layers.CudaConvnetPooling2DLayer(last_layer, pool_size=int(layer['kernel_size']), stride=int(layer['stride']))
	return pool

def ip_layer_from_params(layer, last_layer):
	dense = layers.DenseLayer(last_layer, int(layer['num_output']), -1, -1, nonlinearity=layers.identity) #dropout default 0
	return dense

def dropout_layer_from_params(layer, last_layer):
	'''
	TODO: IMPLEMENT THIS. currently only using this script for forward passes, so this can be a complete identity
	but in the future maybe i'll want to finetune, so this would need to be implemented.
	'''
	return layers.IdentityLayer(last_layer)

def softmax_layer_from_params(layer, last_layer):
	return layers.SoftmaxLayer(last_layer)

if __name__ == '__main__':
	forward, net, all_layers = convert('/root/caffe/models/bvlc_reference_rcnn_ilsvrc13/deploy.prototxt','/root/caffe/models/bvlc_reference_rcnn_ilsvrc13/bvlc_reference_rcnn_ilsvrc13.caffemodel')
