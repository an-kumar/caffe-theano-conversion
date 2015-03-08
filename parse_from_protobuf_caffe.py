def parse_caffemodel(prototxt, caffemodel):
	'''
	parses the caffemodel file using caffe
	'''
	net = caffe.Net(prototxt, caffemodel, caffe.TEST)
	param_dict= net.params
	return param_dict
