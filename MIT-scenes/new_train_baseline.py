from solver import *
from lasagne import layers, nonlinearities,objectives, updates, init, utils, regularization
import theano
import theano.tensor as T
import sys
sys.path.insert(0,'../')
from caffe2theano import models
import numpy as np

'''
SCORES: FC6 LogisticRegression: 0.56315789473684208
		FC7 LogisticRegression: 0.55263157894736847

		SIMPLE ENSEMBLE FC6,FC7: 0.58120300751879694

		SIMPLE AVERAGE FC6, FC7: 0.5759398496240602

		GATED AVERAGE: .599 !!

'''
class Eye(init.Initializer):
	def __init__(self, scale=1):
		self.scale=scale

	def sample(self, shape):
		assert shape[0] == shape[1]
		assert len(shape) == 2
		return utils.floatX(np.eye(shape[0]) * self.scale)

class GatedMultipleInputsLayer(layers.MultipleInputsLayer):
	'''
	A layer that takes in multiple inputs *of the same dimensionality* and computes gates to combine them
	'''
	def __init__(self, incomings, Ws=init.Uniform(), bs = init.Constant(0.), nonlinearity=nonlinearities.sigmoid, **kwargs):
		super(GatedMultipleInputsLayer,self).__init__(incomings,**kwargs)
		num_out = self.input_shapes[0][1]
		# make gates
		self.Ws = [self.create_param(Ws, (num_out,num_out)) for i in range(len(incomings))]
		self.bs = [self.create_param(bs, (num_out,)) for i in range(len(incomings))]

		self.num_inputs = len(incomings)
		self.nonlinearity = nonlinearity


	def get_output_for(self, inputs, *args, **kwargs):
		# compute gates
		gs = [self.nonlinearity(T.dot(inputs[i], self.Ws[i]) + self.bs[i].dimshuffle('x',0)) for i in range(self.num_inputs)]
		# hadamard product
		new_inps = [gs[i] * inputs[i] for i in range(self.num_inputs)]
		# stack into one tensor
		tens = T.stack(*new_inps)
		# now average
		return T.mean(tens, axis=0)

	def get_params(self):
		return self.Ws + self.bs

	def get_output_shape_for(self, input_shapes):
		# assert that the input shapes are the same
		assert len(set(input_shapes)) == 1
		# output is the same as input
		return input_shapes[0]

	def get_output(self, inputs, *args, **kwargs):
		'''
		overwrite the get_output function, this probably could be changed in lasagne.
		'''
		layer_inputs = [self.input_layers[i].get_output(inputs[i], *args, **kwargs) for i in range(self.num_inputs)]
		return self.get_output_for(layer_inputs,*args,**kwargs)

LEARNING_RATE =0.008
MOMENTUM=0.9
REG = .01
solv = SGDMomentumSolver(LEARNING_RATE)
batch_size = 50
# == defining model == # 
print "build model.."
input_one = layers.InputLayer((50, 4096))
input_two = layers.InputLayer((50,4096))
gated_avg = GatedMultipleInputsLayer([input_one,input_two])
output = layers.DenseLayer(gated_avg, num_units=67, nonlinearity=nonlinearities.softmax)
lmodel = models.BaseModel(output)
print "load datasets.."

# load datasets
X_train_fc6 = np.load('/root/proj/MIT_dumped/X_train_fc6.npy')
X_test_fc6 = np.load('/root/proj/MIT_dumped/X_test_fc6.npy')
y_train = np.load('/root/proj/MIT_dumped/y_train.npy')
y_test = np.load('/root/proj/MIT_dumped/y_test.npy')
# load datasets
X_train_fc7 = np.load('/root/proj/MIT_dumped/X_train_fc7.npy')
X_test_fc7 = np.load('/root/proj/MIT_dumped/X_test_fc7.npy')
ds = MultipleInputDataset([X_train_fc6,X_train_fc7], y_train, [X_test_fc6, X_test_fc7], y_test)
# make pred func (todo: put in model)
pred = T.argmax(
    lmodel.get_output(ds.X_batch_var, deterministic=True), axis=1)
lmodel.pred_func = T.mean(T.eq(pred, ds.y_batch_var), dtype=theano.config.floatX)

num_epochs= 1000
solver.solve(lmodel, ds, batch_size, num_epochs)