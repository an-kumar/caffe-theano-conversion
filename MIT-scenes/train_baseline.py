from lasagne import layers, nonlinearities,objectives, updates, init, utils, regularization
import theano
import theano.tensor as T
import sys
import numpy as np

'''
SCORES: FC6 LogisticRegression: 0.56315789473684208
        FC7 LogisticRegression: 0.55263157894736847

        SIMPLE ENSEMBLE FC6,FC7: 0.58120300751879694

        SIMPLE AVERAGE FC6, FC7: 0.5759398496240602

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
    def __init__(self, incomings, Ws=init.Uniform(), bs = init.Constant(0.), nonlinearity=nonlinearities.sigmoid, prob_func=nonlinearities.linear, **kwargs):
        super(GatedMultipleInputsLayer,self).__init__(incomings,**kwargs)
        num_out = self.input_shapes[0][1]
        # make gates
        self.Ws = [self.create_param(Ws, (num_out,num_out)) for i in range(len(incomings))]
        self.bs = [self.create_param(bs, (num_out,)) for i in range(len(incomings))]

        self.num_inputs = len(incomings)
        self.nonlinearity = nonlinearity
        self.prob_func = prob_func


    def get_output_for(self, inputs, *args, **kwargs):
        # compute gates
        gs = [self.nonlinearity(T.dot(inputs[i], self.Ws[i]) + self.bs[i].dimshuffle('x',0)) for i in range(self.num_inputs)]
    #                gs[0].reshape((1, 1))
        # gs is a list of batch_size x num_outputs
        # turn gates to probabilities
        # stack first, so num_inputs x batch_size x num_outputs
        
        tens_gates = T.stack(*gs)
        # turn into batch_size*num_outputs x num_inputs so that softmax working row wise does what we want
        tens_gates = tens_gates.flatten(2).transpose()
        tens_gates = self.prob_func(tens_gates).transpose()
        # now go back
        gs=T.reshape(tens_gates, (self.num_inputs, inputs[0].shape[0], inputs[0].shape[1]))
        # now hadamard product
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


batch_size = 50
total = 5350
X_batch_one = T.matrix()
X_batch_two = T.matrix()
y_batch = T.ivector()
batch_index = T.iscalar()

input_one = layers.InputLayer((50, 4096))
input_two = layers.InputLayer((50,4096))
gated_avg = GatedMultipleInputsLayer([input_one,input_two], nonlinearity=nonlinearities.tanh, prob_func=nonlinearities.softmax)
output = layers.DenseLayer(gated_avg, num_units=67, nonlinearity=nonlinearities.softmax)


# load datasets
X_train_fc6 = theano.shared(np.load('/root/proj/MIT_dumped/X_train_fc6.npy').astype(theano.config.floatX))
X_test_fc6 = theano.shared(np.load('/root/proj/MIT_dumped/X_test_fc6.npy').astype(theano.config.floatX))
y_train = T.cast(theano.shared(np.load('/root/proj/MIT_dumped/y_train.npy')),'int32')
y_test = T.cast(theano.shared(np.load('/root/proj/MIT_dumped/y_test.npy')),'int32')
# load datasets
X_train_fc7 = theano.shared(np.load('/root/proj/MIT_dumped/X_train_fc7.npy').astype(theano.config.floatX))
X_test_fc7 = theano.shared(np.load('/root/proj/MIT_dumped/X_test_fc7.npy').astype(theano.config.floatX))

all_params = layers.get_all_params(output)

objective = objectives.Objective(output,loss_function=objectives.multinomial_nll)
loss_train = objective.get_loss([X_batch_one, X_batch_two], target=y_batch)


LEARNING_RATE =0.122
MOMENTUM=0.9
REG = .0009
reg_loss = regularization.l2(output) * REG
total_loss = loss_train + reg_loss
upds = updates.nesterov_momentum(total_loss, all_params, LEARNING_RATE, MOMENTUM)
pred = T.argmax(
    output.get_output([X_batch_one, X_batch_two], deterministic=True), axis=1)
accuracy = T.mean(T.eq(pred, y_batch), dtype=theano.config.floatX)

print "begin compiling"
givens =    {X_batch_one: X_train_fc6[batch_index*batch_size:(batch_index+1)*batch_size],
            X_batch_two: X_train_fc7[batch_index*batch_size:(batch_index+1)*batch_size],
            y_batch: y_train[batch_index*batch_size:(batch_index+1)*batch_size]}
train = theano.function([batch_index], loss_train, updates=upds, givens=givens)
test = theano.function([], accuracy, givens={X_batch_one:X_test_fc6, X_batch_two:X_test_fc7, y_batch:y_test})
num_epochs = 1000
for epoch in range(num_epochs):
        print "epoch %s" % epoch
    for batch in range(total/batch_size):
        loss = train(batch)
    if epoch % 25 == 0:
        print test()


print test()




# l2 = layers.DenseLayer(l1, num_units=67, nonlinearity=nonlinearities.softmax)
# objective = objectives.Objective(l2,loss_function=objectives.multinomial_nll)
# loss_train = objective.get_loss(X_batch, target=y_batch)

# # load datasets
# X_train_fc6 = theano.shared(np.load('/root/proj/MIT_dumped/X_train_fc6.npy').astype(theano.config.floatX))
# X_test_fc6 = theano.shared(np.load('/root/proj/MIT_dumped/X_test_fc6.npy').astype(theano.config.floatX))
# y_train = T.cast(theano.shared(np.load('/root/proj/MIT_dumped/y_train.npy')),'int32')
# y_test = T.cast(theano.shared(np.load('/root/proj/MIT_dumped/y_test.npy')),'int32')

# all_params = layers.get_all_params(l2)

# LEARNING_RATE =0.008
# MOMENTUM=0.9
# upds = updates.nesterov_momentum(loss_train, all_params, LEARNING_RATE, MOMENTUM)
# pred = T.argmax(
#     l2.get_output(X_batch, deterministic=True), axis=1)
# accuracy = T.mean(T.eq(pred, y_batch), dtype=theano.config.floatX)

# print "begin compiling fc6..."
# train = theano.function([batch_index], loss_train, updates=upds, givens={X_batch:X_train_fc6[batch_index*batch_size:(batch_index+1)*batch_size], y_batch:y_train[batch_index*batch_size:(batch_index+1)*batch_size]})
# num_epochs = 100
# for epoch in range(num_epochs):
#         print "epoch %s" % epoch
#   for batch in range(total/batch_size):
#       loss = train(batch)

# test = theano.function([], accuracy, givens={X_batch:X_test_fc6, y_batch:y_test})
# print test()




# batch_size = 50
# total = 5350
# X_batch = T.matrix()
# y_batch = T.ivector()
# batch_index = T.iscalar()

# l1 = layers.InputLayer((100, 4096))
# l2 = layers.DenseLayer(l1, num_units=67, nonlinearity=nonlinearities.softmax)
# objective = objectives.Objective(l2,loss_function=objectives.multinomial_nll)
# loss_train = objective.get_loss(X_batch, target=y_batch)

# # load datasets
# X_train_fc7 = theano.shared(np.load('/root/proj/MIT_dumped/X_train_fc7.npy').astype(theano.config.floatX))
# X_test_fc7 = theano.shared(np.load('/root/proj/MIT_dumped/X_test_fc7.npy').astype(theano.config.floatX))
# y_train = T.cast(theano.shared(np.load('/root/proj/MIT_dumped/y_train.npy')),'int32')
# y_test = T.cast(theano.shared(np.load('/root/proj/MIT_dumped/y_test.npy')),'int32')

# all_params = layers.get_all_params(l2)

# LEARNING_RATE =0.05
# MOMENTUM=0.9
# upds = updates.nesterov_momentum(loss_train, all_params, LEARNING_RATE, MOMENTUM)
# pred = T.argmax(
#     l2.get_output(X_batch, deterministic=True), axis=1)
# accuracy = T.mean(T.eq(pred, y_batch), dtype=theano.config.floatX)

# print "begin compiling fc7..."
# train = theano.function([batch_index], loss_train, updates=upds, givens={X_batch:X_train_fc7[batch_index*batch_size:(batch_index+1)*batch_size], y_batch:y_train[batch_index*batch_size:(batch_index+1)*batch_size]})
# num_epochs = 100
# for epoch in range(num_epochs):
#         print "epoch %s" % epoch
#   for batch in range(total/batch_size):
#       loss = train(batch)

# test = theano.function([], accuracy, givens={X_batch:X_test_fc7, y_batch:y_test})
# print test()
