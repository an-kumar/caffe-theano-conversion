from google.protobuf import text_format
import caffe_pb2


valid_convolution = set(['convolution'])
valid_inner_product = set(['innerproduct', 'inner_product'])
valid_relu = set(['relu'])
valid_lrn = set(['lrn'])
valid_pooling = set(['pooling'])
valid_softmax = set(['softmax'])
valid_dropout = set(['dropout'])

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

def parse_model_def(filepath):
	'''
	parses the model definition .prototxt file.

	filepath: /path/to/model-def.prototxt

	returns: (input_dimensions, list_of_layers)
	'''
	# open the file and read it's contents
	f = open(filepath)
	contents = f.read()
	f.close()

	# make a netparameter object
	netparam = caffe_pb2.NetParameter()
	# read the contents into the net parameters
	text_format.Merge(contents, netparam)

	# now go through and turn the netparam into a list of dictionaries storing layers
	# this is done to standardize across ".layer", ".layers", ".input_shape", ".input_dim", etc.
	input_dims = find_input_dims(netparam)
	# now we get all layers
	layers = find_layers(netparam)

	name = netparam.name
	inp = (netparam.input[0], input_dims)

	return name, inp, layers



def parse_caffemodel(filepath):
	'''
	parses the trained .caffemodel file

	filepath: /path/to/trained-model.caffemodel

	returns: a dict mapping layer name -> layer blobs
	'''
	f = open(filepath)
	contents = f.read()

	netparam = caffe_pb2.NetParameter()
	netparam.ParseFromString(contents)

	layers = find_layers(netparam)
	param_dict = {} # goes name -> parameter. TODO: something else?
	for layer in layers:
		param_dict[layer.name] = layer.blobs

	return param_dict






def parse_architecture_prototxt(filepath):
	'''
	parses the model definition .prototxt file.

	filepath: /path/to/model-def.prototxt

	returns: (input_dimensions, list_of_layers)
	'''
	# open the file and read it's contents
	f = open(filepath)
	contents = f.read()

	# make a netparameter object
	netparam = caffe_pb2.NetParameter()
	# read the contents into the net parameters
	text_format.Merge(contents, netparam)

	# now go through and turn the netparam into a list of dictionaries storing layers
	# this is done to standardize across ".layer", ".layers", ".input_shape", ".input_dim", etc.
	input_dims = find_input_dims(netparam)
	# now we get all layers
	print "parsing layers..."
	layers = []
	for layer in find_layers(netparam):
		layers.append(layer_dict_from_layer(layer))
	
	# and return
	return input_dims, layers


def parse_model_def(filepath):
	'''
	parses the model definition .prototxt file.

	filepath: /path/to/model-def.prototxt

	returns: (input_dimensions, list_of_layers)
	'''
	# open the file and read it's contents
	f = open(filepath)
	contents = f.read()

	# make a netparameter object
	netparam = caffe_pb2.NetParameter()
	# read the contents into the net parameters
	text_format.Merge(contents, netparam)

	# now go through and turn the netparam into a list of dictionaries storing layers
	# this is done to standardize across ".layer", ".layers", ".input_shape", ".input_dim", etc.
	input_dims = find_input_dims(netparam)
	# now we get all layers
	layers = find_layers(netparam)

	name = netparam.name
	inp = (netparam.input[0], input_dims)
	return name, inp, layers





def layer_dict_from_layer(layer):
	ldict = {}
	ltype = find_layer_type(layer)
	ldict['type'] = ltype
	ldict['name'] = layer.name


	if ltype in valid_convolution:
		print "found conv"
		ldict['num_output'] = layer.convolution_param.num_output
		ldict['kernel_size'] = layer.convolution_param.kernel_size
		ldict['weight_filler'] = layer.convolution_param.weight_filler
		ldict['bias_term'] = layer.convolution_param.bias_term
		ldict['stride'] = layer.convolution_param.stride
		ldict['group'] = layer.convolution_param.group

	elif ltype in valid_pooling:
		print "found pool"
		ldict['kernel_size'] = layer.pooling_param.kernel_size
		ldict['pool'] = layer.pooling_param.pool
		ldict['pad'] = layer.pooling_param.pad
		ldict['stride'] = layer.pooling_param.stride

	elif ltype in valid_lrn:
		print "found lrn"
		ldict['local_size'] = layer.lrn_param.local_size
		ldict['alpha'] = layer.lrn_param.alpha
		ldict['beta'] = layer.lrn_param.beta
		ldict['norm_region'] = layer.lrn_param.norm_region

	elif ltype in valid_inner_product:
		print "found inner product"
		ldict['num_output'] = layer.inner_product_param.num_output
		ldict['weight_filler'] = layer.inner_product_param.weight_filler
		ldict['bias_filler'] = layer.inner_product_param.bias_filler
		ldict['bias_term'] = layer.inner_product_param.bias_term

	elif ltype in valid_dropout:
		print "found dropout"
		ldict['dropout_ratio'] = layer.dropout_param.dropout_ratio

	elif ltype in valid_relu:
		print "found relu"
		ldict['negative_slope'] = layer.relu_param.negative_slope

	elif ltype in valid_softmax:
		print "found softmax"
	else:
		raise Exception("That layer type (%s) is not currently supported" % ltype)

	return ldict

def find_layer_type(layer):
	'''
	checks if the netparam is V1 and if so maps the type

	RETURNS IN LOWER CASE!
	'''
	if type(layer) == caffe_pb2.V1LayerParameter:
		return V1Map[layer.type].lower()


def find_layers(netparam):
	if len(netparam.layers) > 0:
		return netparam.layers
	elif len(netparam.layer) > 0:
		return netparam.layer
	else:
		raise Exception ("Couldn't find layers")


def find_input_dims(netparam):
	'''
	checks netparam.input_dims and .input_shape to find which stores the shape
	'''
	if len(netparam.input_dim) > 0:
		return netparam.input_dim
	elif len(netparam.input_shape) > 0:
		return netparam.input_shape
	else:
		raise Exception("Couldn't find input dimensions in the NetParameter object")
