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
	returns the BaseModel from the given fp
	'''
	params, last_layer, compile_kwargs = pkl.load(open(fp))
	model = BaseModel(last_layer)
	layers.set_all_param_values(model.last_layer, params)
	model.compile(**compile_kwargs)
	return model




class BaseModel(object):
	def __init__(self, last_layer, compile_kwargs={}):
		# get all the layers
		self.all_layers = layers.get_all_layers(last_layer)
		# save input, last layer
		# currently assumed that all_layers[-1] will be input (this is how it should be, i think, but edge cases might exist)
		self.last_layer = last_layer
		self.input_layer = self.all_layers[-1]
		self.compile_kwargs = compile_kwargs
		self.compile(**compile_kwargs)


	def compile(self, *args, **kwargs):
		'''
		This should be overwritten by most models that superclass BaseModel.

		Currently, this just compiles the forward pass (used for transfer learning without finetuning)
		'''
		self.compile_kwargs = kwargs
		nOutputs = kwargs.get('nOutputs', 1)
		# get symbolic input from layer
		symbolic_input = self.input_layer.input_var
		# make list of outputs
		outputs = [lay.get_output() for lay in self.all_layers[-nOutputs:]]
		# store function in self.forward
		self.forward = theano.function([symbolic_input], outputs)






