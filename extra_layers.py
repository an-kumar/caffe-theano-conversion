'''
defines some extra layers in lasagne form to use to convert caffe models
'''

import numpy as np
import theano
import theano.tensor as T

import lasagne.layers as layers
import lasagne.nonlinearities as nonlinearities
import lasagne.init as init


class CaffeConv2DLayer(layers.Conv2DLayer):
    def __init__(self, incoming, num_filters, filter_size, group=1, strides=(1, 1), border_mode="valid", untie_biases=False, W=init.Uniform(), b=init.Constant(0.), nonlinearity=nonlinearities.rectify,convolution=T.nnet.conv2d, **kwargs):
        self.group= group
        super(CaffeConv2DLayer,self).__init__(incoming, num_filters, filter_size, strides=(1, 1), border_mode="valid", untie_biases=False, W=init.Uniform(), b=init.Constant(0.), nonlinearity=nonlinearities.rectify,convolution=T.nnet.conv2d, **kwargs)

    def get_W_shape(self):
        num_input_channels = self.input_shape[1]
        assert num_input_channels % self.group == 0
        return (self.num_filters/self.group, num_input_channels/self.group, self.filter_size[0], self.filter_size[1])

    def get_output_for(self, input, input_shape=None, *args, **kwargs):
        # the optional input_shape argument is for when get_output_for is
        # called directly with a different shape than self.input_shape.
        if input_shape is None:
            input_shape = (self.input_shape[0],self.input_shape[1]/self.group,self.input_shape[2], self.input_shape[3])

        filter_shape = self.get_W_shape()

        if self.border_mode in ['valid', 'full']:
            tensors=[]
            for g in range(self.group):
                inp = input[:,g*(filter_shape[1]/self.group):(g+1)*(filter_shape[1]/self.group),:,:]
                tensors.append(self.convolution(inp, self.W[g*(self.num_filters/2):(g+1)*(self.num_filters/2),:,:,:], subsample=self.strides,
                                      image_shape=input_shape,
                                      filter_shape=filter_shape,
                                      border_mode=self.border_mode))
            conved = T.concatenate(tensors, axis=1)

        elif self.border_mode == 'same':
            tensors=[]
            for g in range(self.group):
                inp = input[:,g*(filter_shape[1]/self.group):(g+1)*(filter_shape[1]/self.group),:,:]
                tensors.append(self.convolution(inp, self.W[self.num_filters/2,:,:,:], subsample=self.strides,
                                      image_shape=input_shape,
                                      filter_shape=filter_shape,
                                      border_mode=self.border_mode))
            conved = T.concatenate(tensors, axis=1)
            shift_x = (self.filter_size[0] - 1) // 2
            shift_y = (self.filter_size[1] - 1) // 2
            conved = conved[:, :, shift_x:input_shape[2] + shift_x,
                            shift_y:input_shape[3] + shift_y]
        else:
            raise RuntimeError("Invalid border mode: '%s'" % self.border_mode)

        if self.b is None:
            activation = conved
        elif self.untie_biases:
            activation = conved + self.b.dimshuffle('x', 0, 1, 2)
        else:
            activation = conved + self.b.dimshuffle('x', 0, 'x', 'x')

        return self.nonlinearity(activation)

       




class SoftmaxLayer(layers.Layer):
    def __init__(self, incoming, **kwargs):
        super(SoftmaxLayer,self).__init__(incoming, **kwargs)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, *args, **kwargs):
        return T.nnet.softmax(input)

class IdentityLayer(layers.Layer):
    def __init__(self, incoming, **kwargs):
        super(IdentityLayer,self).__init__(incoming, **kwargs)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, *args, **kwargs):
        return input


# very heavily copied from lasagne's own dense layer, but caffe does everything backwards so this needs to
# be slightly changed
class CaffeDenseLayer(layers.Layer):
    def __init__(self, incoming, num_units, W=init.Uniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 **kwargs):
        super(CaffeDenseLayer, self).__init__(incoming, **kwargs)
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
            input = input[:,:,::-1,::-1].flatten(2)

        activation = T.dot(input, self.W)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)


class ReluLayer(layers.Layer):
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