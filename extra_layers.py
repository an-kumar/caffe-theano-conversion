'''
defines some extra layers in lasagne form to use to convert caffe models
'''

import numpy as np
import theano
import theano.tensor as T

import lasagne.layers as layers
import lasagne.nonlinearities as nonlinearities



class SoftmaxLayer(layers.layer):
	def __init__(self, incoming, **kwargs):
		super(SoftmaxLayer,self).__init__(incoming, **kwargs)

	def get_output_shape_for(self, input_shape):
		return input_shape

	def get_output_for(self, input, *args, **kwargs):
		return T.nnet.softmax(input)

class IdentityLayer(layers.layer):
	def __init__(self, incoming, **kwargs):
		super(IdentityLayer,self).__init__(incoming, **kwargs)

	def get_output_shape_for(self, input_shape):
		return input_shape

	def get_output_for(self, input, *args, **kwargs):
		return input


# very heavily copied from lasagne's own dense layer, but caffe does everything backwards so this needs to
# be slightly changed
class CaffeDenseLayer(layers.layer):
"""
A fully connected layer.

:parameters:
    - input_layer : `Layer` instance
        The layer from which this layer will obtain its input

    - num_units : int
        The number of units of the layer

    - W : Theano shared variable, numpy array or callable
        An initializer for the weights of the layer. If a Theano shared
        variable is provided, it is used unchanged. If a numpy array is
        provided, a shared variable is created and initialized with the
        array. If a callable is provided, a shared variable is created and
        the callable is called with the desired shape to generate suitable
        initial values. The variable is then initialized with those values.

    - b : Theano shared variable, numpy array, callable or None
        An initializer for the biases of the layer. If a Theano shared
        variable is provided, it is used unchanged. If a numpy array is
        provided, a shared variable is created and initialized with the
        array. If a callable is provided, a shared variable is created and
        the callable is called with the desired shape to generate suitable
        initial values. The variable is then initialized with those values.

        If None is provided, the layer will have no biases.

    - nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.

:usage:
    >>> from lasagne.layers import InputLayer, DenseLayer
    >>> l_in = InputLayer((100, 20))
    >>> l1 = DenseLayer(l_in, num_units=50)
"""
def __init__(self, incoming, num_units, W=init.Uniform(),
             b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
             **kwargs):
    super(DenseLayer, self).__init__(incoming, **kwargs)
    if nonlinearity is None:
        self.nonlinearity = nonlinearities.identity
    else:
        self.nonlinearity = nonlinearity

    self.num_units = num_units

    num_inputs = int(np.prod(self.input_shape[1:]))

    self.W = self.create_param(W, (num_inputs, num_units), name="W")
    self.b = (self.create_param(b, (num_units,), name="b")
              if b is not None else None)

def get_params(self):
    return [self.W] + self.get_bias_params()

def get_bias_params(self):
    return [self.b] if self.b is not None else []

def get_output_shape_for(self, input_shape):
    return (input_shape[0], self.num_units)

def get_output_for(self, input, *args, **kwargs):
    if input.ndim > 2:
        # if the input has more than two dimensions, flatten it into a
        # batch of feature vectors.
        # caffe basically flips all the filters, so we need to reverse some stuff to get this right.
        # i think in gpu mode we won't need to do this
        input = input[:,:,::-1,::-1].flatten(2)[:,:,::-1,::-1]

    activation = T.dot(input, self.W)
    if self.b is not None:
        activation = activation + self.b.dimshuffle('x', 0)
    return self.nonlinearity(activation)


class ReluLayer(layers.layer):
	def __init__(self, incoming, **kwargs):
		super(ReluLayer,self).__init__(incoming, **kwargs)

	def get_output_shape_for(self, input_shape):
		return input_shape

	def get_output_for(self, input, *args, **kwargs):
		return nonlinearities.rectify(input)


# TAKEN FROM LASAGNE PULL REQUEST + PYLEARN2
# SEE https://github.com/benanne/Lasagne/pull/132/files
# but I've changed it to be Caffe style (alpha gets divided by n, k set a 1 always

class CaffeLocalResponseNormalization2DLayer(layers.Layer):
    """
    Cross-channel Local Response Normalization for 2D feature maps.

    Aggregation is purely across channels, not within channels,
    and performed "pixelwise".

    Input order is assumed to be `BC01`.

    If the value of the ith channel is :math:`x_i`, the output is

    .. math::

        x_i = \frac{x_i}{ (k + ( \alpha \sum_j x_j^2 ))^\beta }

    where the summation is performed over this position on :math:`n`
    neighboring channels.

    This code is adapted from pylearn2.
    """

    def __init__(self, incoming, alpha=1e-4, beta=0.75, n=5, **kwargs):
        """
        :parameters:
            - incoming: input layer or shape
            - alpha: see equation above
            - k: see equation above
            - beta: see equation above
            - n: number of adjacent channels to normalize over.
        """
        super(CaffeLocalResponseNormalization2DLayer, self).__init__(incoming, **kwargs)
        self.alpha = alpha
        self.k = 1 # caffe style
        self.beta = beta
        self.n = n
        if n % 2 == 0:
            raise NotImplementedError("Only works with odd n")

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, *args, **kwargs):
        input_shape = self.input_shape
        if any(s is None for s in input_shape):
            input_shape = input.shape
        half_n = self.n // 2
        input_sqr = T.sqr(input)
        b, ch, r, c = input_shape
        extra_channels = T.alloc(0., b, ch + 2*half_n, r, c)
        input_sqr = T.set_subtensor(extra_channels[:,half_n:half_n+ch,:,:], input_sqr)
        scale = self.k
        for i in range(self.n):
            scale += (self.alpha/self.n) * input_sqr[:,i:i+ch,:,:]
        scale = scale ** self.beta
        return input / scale