import caffe
def parse_caffemodel(caffemodel, prototxt):
	'''
	parses the caffemodel file using caffe
	'''
	net = caffe.Net(prototxt, caffemodel, caffe.TEST)
	param_dict= net.params
	return param_dict
