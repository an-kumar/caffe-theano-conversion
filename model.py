import lasagne
import lasagne.layers as layers
import theano
import theano.tensor as T
import cPickle as pkl
def dump(model, fp):
	'''
	dumps the model to the given file path
	'''
	params = layers.get_all_param_values(model.last_layer)
	pkl.dump((params, model.last_layer, model.compile_kwargs), open(fp,'w'))

def load(fp):
	'''
	returns the lasagnemodel from the given fp
	'''
	params, last_layer, compile_kwargs = pkl.load(open(fp))
	model = LasagneModel(last_layer)
	layers.set_all_param_values(model.last_layer, params)
	model.compile_forward(**compile_kwargs)
	return model




class LasagneModel(object):
	def __init__(self, last_layer):
		# get all the layers
		self.all_layers = layers.get_all_layers(last_layer)
		# save input, last layer
		# currently assumed that all_layers[-1] will be input (this is how it should be, i think, but edge cases might exist)
		self.last_layer = last_layer
		self.input_layer = self.all_layers[-1]
		self.compile_kwargs = {}


	def compile_forward(self, *args, **kwargs):
		'''
		outputs is the number of outputs to return (starting from the last layer)

		this should be extended to take a list of layer names instead (or in addition)

		TODO: not sure why people always make the input to the theano function an index rather than index the dataset and just input that, i.e:
		f=theano.function([idx], output, givens={self.input_layer.input_var:ds[idx]})
		for idx in __:
			f(idx)
		vs
		f = theano.function([self.input_layer.input_var],output)
		for idx in __:
			f(ds[idx])

		maybe do some benchmarking and see which is faster (could be how shared datasets are used), but for now this is cleaner
		'''
		self.compile_kwargs = kwargs
		nOutputs = kwargs.get('nOutputs', 1)
		# get symbolic input from layer
		symbolic_input = self.input_layer.input_var
		# make list of outputs
		outputs = [lay.get_output() for lay in self.all_layers[-nOutputs:]]
		# store function in self.forward
		self.forward = theano.function([symbolic_input], outputs)






