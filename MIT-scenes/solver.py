import theano
import theano.tensor as T
import numpy as np
import lasagne
'''
usage should be:
solver.solve(model, dataset, batch_size)

for now, dataset is a shared variable (later turned into object of a ds class)
'''

def tensortype_from_shape(shape, intg=False):
	if len(shape) == 1:
		''' HACK!:'''
		if intg:
			return T.ivector
		else:
			return T.vector
	elif len(shape) == 2:
		return T.matrix
	elif len(shape) == 3:
		return T.tensor3
	elif len(shape) == 4:
		return T.tensor4




class BaseSolver(object):
	def __init__(self, reg='l2', reg_scale=0.01):
		self.reg = reg
		self.reg_scale = np.cast[theano.config.floatX](0.01)
		self.lasagne_reg = self.find_lasagne_reg()

	def get_reg_loss(self, model, include_bias=False):
		# all_params = self.get_all_params(model, include_bias=include_bias)
		loss = self.lasagne_reg(model.last_layer)
		print loss.type
		return loss * self.reg_scale


	def find_objective(self, model, objective):
		if objective == 'multinomial_nll':
			return lasagne.objectives.Objective(model.last_layer, lasagne.objectives.multinomial_nll)


	def find_lasagne_reg(self):
		if self.reg == 'l2':
			return lasagne.regularization.l2

	def get_all_params(self, model, include_bias=True):
		'''
		finds all the parameters of a lasagne model
		'''
		if include_bias:
			return lasagne.layers.get_all_params(model.last_layer)
		else:
			return lasagne.layers.get_all_non_bias_params(model.last_layer)

	def get_updates(self):
		raise NotImplementedError


class MultipleInputDataset(object):
	'''
	for now, extremely simple. todo: decompose to BaseDataset
	'''
	def __init__(self, X_trains, y_train, X_tests, y_test, y_cast='int32'):
		'''
		all args are numpy format
		'''
		self.X_trains = [theano.shared(X_train.astype(theano.config.floatX)) for X_train in X_trains]
		self.y_train = T.cast(theano.shared(y_train.astype(theano.config.floatX)),y_cast)
		self.X_tests = [theano.shared(X_test.astype(theano.config.floatX)) for X_test in X_tests]
		self.y_test = T.cast(theano.shared(y_test.astype(theano.config.floatX)),y_cast)

		self.X_batch_var = self.get_X_batch_var(X_trains)
		self.y_batch_var = self.get_y_batch_var(y_train)

		self.train_size = y_train.shape[0]

	def get_X_batch_var(self, X_trains):
		return [tensortype_from_shape(X_train.shape)() for X_train in X_trains]
	def get_y_batch_var(self, y_train):
		return tensortype_from_shape(y_train.shape, intg=True)()

	def train_givens(self, batch_index, batch_size):
		'''
		batch_index is a theano variable
		'''
		givens = {self.X_batch_var[i]:self.X_trains[i][batch_index*batch_size:(batch_index+1)*batch_size] for i in range(len(self.X_trains)) }
		givens[self.y_batch_var] = self.y_train[batch_index*batch_size:(batch_index+1)*batch_size]
		return givens

	def test_givens(self):
		givens = {self.X_batch_var[i]:self.X_tests[i] for i in range(len(self.X_tests))}
		givens[self.y_batch_var] = self.y_test
		return givens		




class SGDMomentumSolver(BaseSolver):
	'''
	TODO: support changing momentum after #epochs
	'''
	def __init__(self, global_lr, momentum=0.9, lr_drop_epochs=None, gamma=0.9, objective='multinomial_nll',**kwargs):
		'''
		global_lr : global learning rate
		momentum : momentum of SGD
		lr_drop_epochs : num epochs before the lr drops
		gamma : after lr_drop_epochs, the lrs become gamma*lr
		'''
		super(SGDMomentumSolver,self).__init__(**kwargs)
		self.global_lr = np.cast[theano.config.floatX](global_lr)
		self.momentum = momentum
		self.specific_W_lrs = {}
		self.specific_b_lrs = {}
		self.objective=objective

	def set_specific_W_lrs(self, lr_dict):
		'''
		sets specific learning rates given layer **names**

		lr_dict : dict mapping layer name -> specific learning rate
		'''
		# iterate through keys to overwrite already set keys
		for key in lr_dict:
			self.specific_W_lrs[key] = np.cast[theano.config.floatX](lr_dict[key])

	def set_specific_b_lrs(self, lr_dict):
		'''
		sets specific learning rate given layer **names**

		lr_dict : dict mapping layer name -> specific learning rate
		'''
		for key in lr_dict:
			self.specific_b_lrs[key] = np.cast[theano.config.floatX](lr_dict[key])

	def set_specific_lrs(self, lr_dict):
		'''
		wraps W and b
		'''
		self.set_specific_b_lrs(lr_dict)
		self.set_specific_W_lrs(lr_dict)

	def solve(self, model, dataset, batch_size, num_epochs, include_bias=True):
		# get the obj_loss, reg_loss
		X_batch = dataset.X_batch_var # could be a list
		y_batch = dataset.y_batch_var # could be a list
		obj = self.find_objective(model, self.objective)
		obj_loss = obj.get_loss(X_batch, target=y_batch)

		all_params = self.get_all_params(model, include_bias=include_bias)
		reg_loss = self.get_reg_loss(model, include_bias=False)

		updates, all_lrs = self.get_updates(obj_loss + reg_loss, all_params)

		# init all lrs to the global lr
		solver_givens = {val:self.global_lr for val in all_lrs.values()}
		# go through and update the ones that have been set specifically
		for name in self.specific_W_lrs:
			# get the correct param
			params = model.get_W_params_by_name(name)
			for param in params:
				solver_givens[all_lrs[param]] = self.specific_W_lrs[name]

		for name in self.specific_b_lrs:
			params = model.get_b_params_by_name(name)
			for param in params:
				solver_givens[all_lrs[param]] = self.specific_b_lrs[name]

		# add dataset givens
		# todo: is "batch_index" not general enough?
		batch_index = T.iscalar('batch')
		
		solver_givens.update(dataset.train_givens(batch_index, batch_size))

		# compile function
		# todo: obj loss?
		
		func = theano.function([batch_index], obj_loss, updates=updates, givens=solver_givens)
		if model.pred_func is not None:
			test_func = theano.function([], model.pred_func, givens=dataset.test_givens())
		else:
			test_func = None
		# train
		batches_per_epoch = int(np.ceil(float(dataset.train_size )/ batch_size))
                test_batches_per_epoch = int(np.ceil(float(dataset.test_size) / batch_size))
		loss_history = []
		for epoch in range(num_epochs):
			for batch in range(batches_per_epoch):
				print "BATCH: %s" % batch
				dataset.deal_with_batch(batch, batch_size)
				loss_history.append(func(batch))
			if epoch % 25 == 0:
                            print "testing"
                            for batch in range(batches_per_epoch):
                                dataset.deal_with_batch(batch, batch_size,mode='test')

				print "curr loss: %s" % (loss_history[-1])
				if test_func is not None:
					print test_func()

                return loss_history



	def get_updates(self, loss, all_params):
                # return lasagne.updates.momentum(loss,all_params, self.global_lr), {}
		all_grads = theano.grad(loss, all_params)
		# all_lrs maps param -> learning rate
		all_lrs = {param:T.scalar('param_%s' % str(param)) for param in all_params}

		updates = []
		for param_i, grad_i in zip(all_params, all_grads):
			mparam_i = theano.shared(np.zeros(param_i.get_value().shape).astype(theano.config.floatX),broadcastable=param_i.broadcastable)
			v = self.momentum * mparam_i - all_lrs[param_i] * grad_i
			updates.append((mparam_i, v))
			updates.append((param_i, param_i + v))

		return updates, all_lrs





