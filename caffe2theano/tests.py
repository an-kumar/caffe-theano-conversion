import conversion
import time
import os
import theano
import numpy as np
from models import *

def printt(string):
	print "====== [ TESTING: %s ] ======" % string
	return "====== [ TESTING: %s ] ======\n" % string

def printe(string):
	print "====== [ ERROR: %s ] ======" % string
	return "====== [ ERROR: %s ] ======\n" % string
def prints(string):
	print '===== [ STATUS: %s ] ======' % string
	return '===== [ STATUS: %s ] ======\n' % string

def main(prototxt, caffemodel):
	test_string = ''
	test_string += printt('Accuracy of conversion. Caffe required.')
	model = None
	try:
		import caffe
		net = caffe.Net(prototxt,caffemodel,caffe.TEST)
		test_string += printt('Accuracy of conversion - caffe parsing')
		model = conversion.convert(prototxt,caffemodel,caffe_parse=True)
		l2_distance = test_similarity(model,net)
		if l2_distance < 1e-7:
			test_string += prints('Accuracy of conversion - caffe parsing: Passed')
		else:
			test_string += prints('Accuracy of conversion - caffe parsing: Failed')

		del model
		test_string += printt('Accuracy of conversion - protobuf parsing')
		model = conversion.convert(prototxt,caffemodel,caffe_parse=False)
		l2_distance = test_similarity(model,net)
		if l2_distance < 1e-7:
			test_string += prints('Accuracy of conversion - protobuf parsing: Passed')
		else:
			test_string += prints('Accuracy of conversion - protobuf parsing: Failed')
	except Exception as e:
		print e
		test_string += printe('Caffe was not found. Continuing...')

	test_string += printt('Serialization')
	if model is None:
		model = conversion.convert(prototxt,caffemodel)
	success = test_serialization(model)
	if success:
		test_string += prints('Serialization: Passed')
	else:
		test_string += prints('Serialization: Failed')

	print '=====================================\n'*10
	print 'SUMMARY:'
	print test_string



def test_similarity(model, net):
	inp_shape= net.blobs['data'].data.shape
	random_mat = np.random.randn(*inp_shape).astype(theano.config.floatX) 
	
        tick = time.time()
        for i in range(10):
	    fprop = net.forward(**{net.inputs[0]:random_mat})
	#print fprop[fprop.keys()[0]].shape
	tock = time.time()
        print "caffe took: %s" % (tock-  tick)
	
	tick = time.time()
        for i in range(10):
	    outlist = model.forward(random_mat)
	tock = time.time()
        print "lasagne took: %s" % (tock - tick)
	print 'model forward'
	
	# print fprop vs outlist
	print 'L2 distance between output of caffe and output of theano'
	print np.sum((fprop[fprop.keys()[0]][:,:,0,0] - outlist[0])**2)
	print 'Max absolute different between entries in caffe and entries in theano'
	print np.amax(np.abs(fprop[fprop.keys()[0]][:,:,0,0]-outlist[0]))

	return np.sum((fprop[fprop.keys()[0]][:,:,0,0] - outlist[0])**2)


def test_serialization(model):
	random_mat = np.random.randn(*(model.input_layer.shape)).astype(theano.config.floatX)
	print "outlist_1"
	outlist_1 = model.forward(random_mat)
	print "dumping..."
	dump(model, 'temp_test.lm')
	print "loading..."
	loaded_model = load('temp_test.lm')
	os.system('rm temp_test.lm')
	print "begin outlist 2"
	outlist_2 = loaded_model.forward(random_mat)

	for i in range(len(outlist_1)):
		print 'L2 Distance between outputs:'
		print np.sum((outlist_1[i] - outlist_2[i])**2)
		if np.sum((outlist_1[i] - outlist_2[i])**2) > 1e-7:
			return False
		print 'Max absolute difference between entries:'
		print np.amax(np.abs(outlist_1[i]-outlist_2[i]))

	return True

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--prototxt", default='../../caffe/models/bvlc_reference_caffenet/deploy.prototxt', help="model definition file")
	parser.add_argument("--caffemodel", default='../../caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',help="model binary")
	args = parser.parse_args()
	main (args.prototxt, args.caffemodel)
