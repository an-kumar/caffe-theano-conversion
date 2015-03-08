import theano
import numpy as np
import time
import theano.tensor as T
import extra_layers
from model import LasagneModel, dump, load
try:
	from lasagne.layers import cuda_convnet
	import extra_convnet_layers
	print "===============\n"*10
	print "using cuda convnet wrappers"
	cuda = True
except Exception as e:
	print e
	print "probably no GPU, or no pylearn2 capabilities; using normal"
	cuda = False
import lasagne.layers as layers
# from parse_model_def import parse_model_def as parse
from parse_from_protobuf import parse_model_def, parse_caffemodel
try:
	import caffe
except:
	print 'continuing without caffe'
import lasagne.nonlinearities as nonlinearities

V1Map = {0: 'NONE',
 1: 'ACCURACY',
 2: 'BNLL',
 3: 'CONCAT',
 4: 'CONVOLUTION',
 5: 'DATA',
 6: 'DROPOUT',
 7: 'EUCLIDEAN_LOSS',
 8: 'FLATTEN',
 9: 'HDF5_DATA',
 10: 'HDF5_OUTPUT',
 11: 'IM2COL',
 12: 'IMAGE_DATA',
 13: 'INFOGAIN_LOSS',
 14: 'INNER_PRODUCT',
 15: 'LRN',
 16: 'MULTINOMIAL_LOGISTIC_LOSS',
 17: 'POOLING',
 18: 'RELU',
 19: 'SIGMOID',
 20: 'SOFTMAX',
 21: 'SOFTMAX_LOSS',
 22: 'SPLIT',
 23: 'TANH',
 24: 'WINDOW_DATA',
 25: 'ELTWISE',
 26: 'POWER',
 27: 'SIGMOID_CROSS_ENTROPY_LOSS',
 28: 'HINGE_LOSS',
 29: 'MEMORY_DATA',
 30: 'ARGMAX',
 31: 'THRESHOLD',
 32: 'DUMMY_DATA',
 33: 'SLICE',
 34: 'MVN',
 35: 'ABSVAL',
 36: 'SILENCE',
 37: 'CONTRASTIVE_LOSS',
 38: 'EXP',
 39: 'DECONVOLUTION'}

# valid names (lowercased)
valid_convolution = set(['convolution'])
valid_inner_product = set(['innerproduct', 'inner_product'])
valid_relu = set(['relu'])
valid_lrn = set(['lrn'])
valid_pooling = set(['pooling'])
valid_softmax = set(['softmax'])
valid_dropout = set(['dropout'])



def convert_model_def(prototxt):
	'''
	prototxt is a model definition .prototxt file

	returns a lasagne model with the layers, correct initialized
	'''
	name, inp, architecture = parse_model_def(prototxt)
	inp_name, inp_dims = inp

	last_layer = inp_layer = layers.InputLayer(tuple(inp_dims), name=inp_name)
	# now go through all layers and create the lasagne equivalent
	for layer in architecture:
		this_layer = parse_layer_from_param(layer, last_layer)
		last_layer = this_layer

	# make the lasagne model (only need last layer)
	model = LasagneModel(last_layer)
	return model

def parse_layer_from_param(layer,last_layer):
	'''
	returns the correct layer given the param dict
	'''
	tp = V1Map[layer.type].lower()

	if tp in valid_convolution:
		if cuda==True:
			return cuda_conv_layer(layer, last_layer)
		else:
			return conv_layer(layer, last_layer)
	elif tp in valid_relu:
		return relu_layer(layer, last_layer)
	elif tp in valid_pooling:
		if cuda==True:
			return cuda_pooling_layer(layer, last_layer)
		else:
			return pooling_layer(layer, last_layer)
	elif tp in valid_inner_product:
		return ip_layer(layer, last_layer)
	elif tp in valid_dropout:
		return dropout_layer(layer, last_layer)
	elif tp in valid_softmax:
		return softmax_layer(layer, last_layer)
	elif tp in valid_lrn:
		return lrn_layer(layer, last_layer)
	else:
		raise Exception('not a valid layer: %s' % tp)



def lrn_layer(layer, last_layer):
	name = layer.name
	param = layer.lrn_param
	# set params
	alpha = param.alpha
	beta = param.beta
	n = param.local_size

	lrn = extra_layers.CaffeLocalResponseNormalization2DLayer(last_layer, alpha=alpha, beta=beta, n=n,name=name)
	return lrn

def cuda_conv_layer(layer, last_layer):
	name = layer.name
	param = layer.convolution_param

	num_filters = param.num_output
	filter_size = (param.kernel_size,param.kernel_size) #only suppose square filters
	strides = (param.stride,param.stride) # can only suport square strides anyways
	## border mode is wierd...
	border_mode = None
	pad = param.pad
	nonlinearity=nonlinearities.identity
	groups= param.group
			
	conv = extra_convnet_layers.CaffeConv2DCCLayer(last_layer, groups=groups, num_filters=num_filters,filter_size=filter_size, strides=strides, border_mode=border_mode, pad=pad, nonlinearity=nonlinearity,name=name)
	return conv

def conv_layer(layer, last_layer):
	name = layer.name
	param = layer.convolution_param
	num_filters = param.num_output
	filter_size = (param.kernel_size, param.kernel_size)
	strides = (param.stride, param.stride)
	group = param.group
	pad = param.pad
	nonlinearity=nonlinearities.identity

	# theano's conv only allows for certain padding, not arbitrary. not sure how it will work if same border mode is not true.
	if (filter_size[0] - pad * 2 ) == 1:
		print "using same convolutions, this should be correct"
		border_mode = 'same'
	elif pad == 0:
		print "using valid border mode, this should work but who knows"
		border_mode='valid'
	elif pad != 0:
		print "pretty sure this won't work but we'll try a full conv"
		border_mode = 'full'
	if group > 1:
		conv = extra_convnet_layers.CaffeConv2DLayer(last_layer, group=group,num_filters=num_filters, filter_size=filter_size, strides=strides, border_mode=border_mode, nonlinearity=nonlinearity,name=name)
	else:
		conv = layers.Conv2DLayer(last_layer, num_filters=num_filters, filter_size=filter_size, strides=strides, border_mode=border_mode, nonlinearity=nonlinearity,name=name)
	return conv

def relu_layer(layer, last_layer):
	name = layer.name
	return extra_layers.ReluLayer(last_layer,name=name)

def pooling_layer(layer, last_layer):
	name = layer.name
	param = layer.pooling_param
	ds=(param.kernel_size,param.kernel_size) #caffe only does square kernels
	strides = (param.stride, param.stride)

	if strides[0] != ds[0]:
		pool = extra_layers.CaffeMaxPool2DLayer(last_layer,ds=ds, strides=strides,name=name)
	else:
		pool = layers.MaxPool2DLayer(last_layer, ds=ds,name=name) # ignore border is set to False, maybe look into how caffe does borders if the strides don't work perfectly
	return pool

def cuda_pooling_layer(layer, last_layer):
	name = layer.name
	param = layer.pooling_param
	ds=(param.kernel_size,param.kernel_size) #caffe only does square kernels
	strides = (param.stride, param.stride)

	pool = cuda_convnet.MaxPool2DCCLayer(last_layer, ds=ds, strides=strides,name=name)
	return pool

def ip_layer(layer, last_layer):
	name = layer.name
	param = layer.inner_product_param
	num_units=param.num_output
	nonlinearity=nonlinearities.identity

	dense = layers.DenseLayer(last_layer, num_units=num_units, nonlinearity=nonlinearity,name=name)
	return dense

def dropout_layer(layer, last_layer):
	name = layer.name
	'''
	TODO: IMPLEMENT THIS. currently only using this script for forward passes, so this can be a complete identity
	but in the future maybe i'll want to finetune, so this would need to be implemented.
	'''
	return extra_layers.IdentityLayer(last_layer, name=name)

def softmax_layer(layer, last_layer):
	name = layer.name
	return extra_layers.SoftmaxLayer(last_layer, name=name)





def set_params_from_caffemodel(lasagne_model, caffemodel):
	'''
	sets the params of lasagne_model to be from the trained caffemodel.

	lasagne_model is a lasagne model from e.g convert_model_def(prototxt).
	caffemodel is the filepath to the .caffemodel file
	'''
	# load in the caffemodel (this takes a long time without cpp implementation of protobuf)
	layer_params = parse_caffemodel(caffemodel)

	# this should be in the same order as was made by the lasagne model, but reversed. we will check that.
	# todo: maybe just go by names, strictly? 
	for layer in layer_params[::-1]:



























def convert(prototxt, caffemodel):
	'''
	prototxt is a .prototxt file
	caffemodel is a .caffemodel file
	'''
	# parse the prototxt file
	input_dims, architecture = parse(prototxt)
	input_dims2, architecture2 = parse2(prototxt)
	return input_dims, architecture, input_dims2, architecture2
	assert len(input_dims) == 4 #bc01
	net = caffe.Net(prototxt, caffemodel, caffe.TEST)

	# create input layer
	# this actually ends up being c01b shaped if cuda, but we pass in bc01
	#if cuda == False:
	last_layer = inp_layer = layers.InputLayer(tuple(input_dims), name='data')
	#else:
		# this actually ends up being c01b shaped if cuda, but we pass in bc01, so we need to reshuffled
	#	last_layer = inp_layer = layers.InputLayer((input_dims[1], input_dims[2], input_dims[3], input_dims[0]), name='data')


	# go thru layers and create the theano layer 
	all_layers = [inp_layer]
	swapped = False
	for layer in architecture:
		# if (layer['type'] == 'INNER_PRODUCT' or layer['type'] == 'InnerProduct') and swapped==False and cuda==True:
		# 	# need to add a reshaping layer
		# 	'''
		# 	this might not be needed, check lasagne stuff
		# 	'''
		# 	reshape_layer = cuda_convnet.ShuffleC01BToBC01Layer(last_layer)
		# 	all_layers.append(reshape_layer)
		# 	last_layer = reshape_layer
		# 	swapped = True

		this_layer = parse_layer(layer, last_layer)
#		if layer['type'].lower() in valid_lrn:
#`			break
		if this_layer == -1:
			# error checking
			continue
		
		set_params(this_layer, net, layer)
		last_layer = this_layer
		all_layers.append(this_layer)
		
	model = LasagneModel(last_layer)
	model.compile_forward(nOutputs=0) # 0 returns all layers
	return model, net, all_layers



def set_params(theano_layer, net, layer_params):
	'''
	theano_layer is a theano layer
	net is the caffe net
	layer_params is the layer params
	'''
	if len(theano_layer.get_params())== 0:
		return # no params to set
	else:
		if layer_params['type'] == 'CONVOLUTION' or layer_params['type'] == 'Convolution':
			if cuda==True:
				set_cuda_conv_params(theano_layer, net, layer_params)
			else:
				set_conv_params(theano_layer, net, layer_params)
		elif layer_params['type'] == 'INNER_PRODUCT' or layer_params['type'] == 'InnerProduct':
			set_ip_params(theano_layer, net, layer_params)
		else:
			print "not a valid layer to set params to (what happened??) %s" % layer_params['type']

def set_conv_params(theano_layer, net, layer_params):
	name = layer_params['name']
	W = net.params[name][0].data
	b = net.params[name][1].data
	# b needs to just be the last index
	b = b[0,0,0,:]
	# W needs to be fixed
	W = W[:,:,::-1,::-1]
	theano_layer.W.set_value(W.astype(theano.config.floatX))
	theano_layer.b.set_value(b.astype(theano.config.floatX))

def set_cuda_conv_params(theano_layer, net, layer_params):
	name = layer_params['name']
	W = net.params[name][0].data
	b = net.params[name][1].data
	# b needs to just be the last index
	b = b[0,0,0,:]
	# W needs to be reshaped into n_features(from prev layer), size, size, n_filters
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
	if layer['type'] == 'CONVOLUTION' or layer['type'] == 'Convolution':
		if cuda==True:
			return cuda_conv_layer_from_params(layer, last_layer)
		else:
			return conv_layer_from_params(layer, last_layer)
	elif layer['type'] == 'RELU' or layer['type'] == 'ReLU':
		return relu_layer_from_params(layer, last_layer)
	elif layer['type'] == 'POOLING' or layer['type'] == 'Pooling':
		if cuda==True:
			return cuda_pooling_layer_from_params(layer, last_layer)
		else:
			return pooling_layer_from_params(layer, last_layer)
	elif layer['type'] == 'INNER_PRODUCT' or layer['type'] == 'InnerProduct':
		return ip_layer_from_params(layer, last_layer)
	elif layer['type'] == 'DROPOUT' or layer['type'] == 'Dropout':
		return dropout_layer_from_params(layer, last_layer)
	elif layer['type'] == 'SOFTMAX' or layer['type'] == 'Softmax':
		return softmax_layer_from_params(layer, last_layer)
	elif layer['type'] == 'LRN':
		return lrn_layer_from_params(layer, last_layer)
	else:
		print 'not a valid layer: %s' % layer['type']
		return -1

def lrn_layer_from_params(layer, last_layer):
	# set params
	alpha = float(layer['alpha'])
	beta = float(layer['beta'])
	n = int(layer['local_size'])

	lrn = extra_layers.CaffeLocalResponseNormalization2DLayer(last_layer, alpha=alpha, beta=beta, n=n)
	return lrn

def cuda_conv_layer_from_params(layer, last_layer):
	num_filters = int(layer['num_output'])
	filter_size = (int(layer['kernel_size']),int(layer['kernel_size'])) #only suppose square filters
	strides = (int(layer['stride']),int(layer['stride'])) # can only suport square strides anyways
	## border mode is wierd...
	border_mode = None
	pad = int(layer['pad'])
	nonlinearity=nonlinearities.identity
	groups= int(layer['group'])
			
	conv = extra_convnet_layers.CaffeConv2DCCLayer(last_layer, groups=groups, num_filters=num_filters,filter_size=filter_size, strides=strides, border_mode=border_mode, pad=pad, nonlinearity=nonlinearity)
	return conv

def conv_layer_from_params(layer, last_layer):
	# theano's conv only allows for certain padding, not arbitrary. not sure how it will work if same border mode is not true.
	if int(layer['kernel_size']) - (int(layer['pad']) * 2 ) == 1:
		print "using same convolutions, this should be correct"
		border_mode = 'same'
	else:
		print "using valid border mode, this should work but who knows"
		border_mode='valid'

	num_filters = int(layer['num_output'])
	filter_size = (int(layer['kernel_size']), int(layer['kernel_size'])) # must be a tuple
	strides = (int(layer['stride']),int(layer['stride'])) # can only suport square strides anyways
	group = int(layer['group'])
	## border mode is wierd...
	
	nonlinearity=nonlinearities.identity

	if group > 1:
		conv = extra_convnet_layers.CaffeConv2DLayer(last_layer, group=group,num_filters=num_filters, filter_size=filter_size, strides=strides, border_mode=border_mode, nonlinearity=nonlinearity)
	else:
		conv = layers.Conv2DLayer(last_layer, num_filters=num_filters, filter_size=filter_size, strides=strides, border_mode=border_mode, nonlinearity=nonlinearity)
	return conv

def relu_layer_from_params(layer, last_layer):
	return extra_layers.ReluLayer(last_layer)

def pooling_layer_from_params(layer, last_layer):
	ds=(int(layer['kernel_size']),int(layer['kernel_size'])) #caffe only does square kernels
	strides = (int(layer['stride']), int(layer['stride']))

	if strides[0] != ds[0]:
		pool = extra_layers.CaffeMaxPool2DLayer(last_layer,ds=ds, strides=strides)
	else:
		pool = layers.MaxPool2DLayer(last_layer, ds=ds) # ignore border is set to False, maybe look into how caffe does borders if the strides don't work perfectly
	return pool

def cuda_pooling_layer_from_params(layer, last_layer):
	ds = (int(layer['kernel_size']),int(layer['kernel_size'])) # cuda only supports square anyways
	strides = (int(layer['stride']), int(layer['stride'])) #only square strides as well

	pool = cuda_convnet.MaxPool2DCCLayer(last_layer, ds=ds, strides=strides)
	return pool

def ip_layer_from_params(layer, last_layer):
	num_units=int(layer['num_output'])
	nonlinearity=nonlinearities.identity
	if cuda==False:
		dense = layers.DenseLayer(last_layer, num_units=num_units, nonlinearity=nonlinearity)
	else:
		dense = layers.DenseLayer(last_layer, num_units=num_units, nonlinearity=nonlinearity)
	return dense

def dropout_layer_from_params(layer, last_layer):
	'''
	TODO: IMPLEMENT THIS. currently only using this script for forward passes, so this can be a complete identity
	but in the future maybe i'll want to finetune, so this would need to be implemented.
	'''
	return extra_layers.IdentityLayer(last_layer)

def softmax_layer_from_params(layer, last_layer):
	return extra_layers.SoftmaxLayer(last_layer)





# ===[ Tests ] ===#

def test_similarity(model, net):
	inp_shape= net.blobs['data'].data.shape
	random_mat = np.random.randn(*inp_shape).astype(theano.config.floatX) #hard coded for VGG ILSVRC 15
	tick = time.time()
	fprop = net.forward(**{net.inputs[0]:random_mat})
	print fprop[fprop.keys()[0]].shape
	tock = time.time()
	print 'time: %s' % str(tock - tick)
	tick = time.time()
	outlist = model.forward(random_mat)
	tock = time.time()
	print 'model forward'
	print 'time: %s' % str(tock - tick)
	# print fprop vs outlist
	print 'L2 distance between output of caffe and output of theano'
	print np.sum((fprop[fprop.keys()[0]][:,:,0,0] - outlist[0])**2)
	print 'Max absolute different between entries in caffe and entries in theano'
	print np.amax(np.abs(fprop[fprop.keys()[0]][:,:,0,0]-outlist[0]))

	return random_mat, outlist


def test_serialization(model,random_mat):
	print "outlist_1"
	outlist_1 = model.forward(random_mat)
	print "dumping..."
	dump(model, 'temp_test.lm')
	print "loading..."
	loaded_model = load('temp_test.lm')
	print "begin outlist 2"
	outlist_2 = loaded_model.forward(random_mat)

	for i in range(len(outlist_1)):
		print 'L2 Distance between outputs:'
		print np.sum((outlist_1[i] - outlist_2[i])**2)
		print 'Max absolute difference between entries:'
		print np.amax(np.abs(outlist_1[i]-outlist_2[i]))


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--prototxt", default='VGG_ILSVRC_16_layers_deploy.prototxt', help="model definition file")
	parser.add_argument("--caffemodel", default='VGG_ILSVRC_16_layers.caffemodel',help="model binary")
	args = parser.parse_args()

	model = convert_model_def(args.prototxt)
	raise

	print 'Converting model...'
	one,two,a,b = convert(args.prototxt,args.caffemodel)
	raise


	model, net, all_layers = convert(args.prototxt,args.caffemodel)
	print 'testing similarity...'
	random_mat, outlist =test_similarity(model, net)
	test_serialization(model, random_mat)
