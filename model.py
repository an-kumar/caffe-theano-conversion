import layers
import cc_layers
import theano
import theano.tensor as T

class conv_model(object):
	def __init__(self, layers):
		self.layers = layers # this should be a list of layers, 
		self.forward, self.backward = self.compile_functions()


	def compile_functions(self):
		'''
		compiles the forward and backward pass of this conv model
		'''
		inp = T
