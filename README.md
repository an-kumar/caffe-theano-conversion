# caffe_theano_conversion

This is a repository that allows you to convert pretrained caffe models into models in Lasagne, a thin wrapper around Theano. You can also convert a caffe model's architecture to an equivalent one in Lasagne. You do not need caffe installed to use this module.

Currently, the following caffe layers are supported:

	* Convolution
	* LRN
	* Pooling
	* Inner Product
	* Relu
	* Softmax

Right now, you have to put the cost layer on yourself, as well as do the backprop code. This is a future step for me, however, Lasagne is very easy to use and you can learn how to add your own stuff very easily. I want to keep this as configurable as possible because that's the benefit of theano.

## DEPENDENCIES:
Theano (http://deeplearning.net/software/theano/), Lasagne (https://github.com/benanne/Lasagne), Google protobuf. Pylearn2 for cuda convnet wrappers (see below). If you have Caffe installed, you already have google protobuf, otherwise see here: https://code.google.com/p/protobuf/

##USING CUDA CONVNET WRAPPERS:

The cuda-convnet wrappers in pylearn2 are much faster than the GPU implementations of convolutions in Theano. Lasagne has cuda-convnet layers, and I have created a caffe version of these layers. However, they require you to go into pylearn2 and change some of the files. I don't know what the best way to package that change in this repo is, so until someone tells me a better way I'll just describe what to do:

In this file: https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/sandbox/cuda_convnet/base_acts.py, add a argument group=1 to the __init__ function of BaseActs. In the __init__ function, change:
```
self.dense_connectivity = True
```
to:
```
if group == 1:
	self.dense_connectivity = True
else:
	self.dense_connectivity = False
```

and add a line:
```
self.group = group
```

then, in this file: https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/sandbox/cuda_convnet/filter_acts.py, change FilterActs c_code function from:

```
if self.dense_connectivity:
	basic_setup += """
    #define numGroups 1
    """
```

to:
```
if self.dense_connectivity:
	basic_setup += """
    #define numGroups 1
    """
else:
	basic_setup += """
	#define numGroups %s
	""" % self.group
```

You should be able to not have that if statement at all, but I kept it in.

## USAGE:
You can test the repo by python tests.py.
All the tests should pass, but **terminal is out of GPUs, so I haven't been able to run the tests.py script on GPU**. However, I have used this repo with GPUs, and it worked, the only question is if it still works after I moved files around. If it doesn't let me know.

The file conversion.py has a function ```convert``` that takes a prototxt file and a caffemodel file, and returns a lasagne base model. I have plans to superclass base models with other models, for the purposes of training, but I haven't yet best figured out how to do this. This repo is still in active development while I use it for my project. If you have ideas for additions, let me know.

usage:
```
>>> from conversion import convert
>>> from models import *
>>> lmodel = convert('/path/to/deploy.prototxt', '/path/to/pretrained_model.caffemodel')
>>> dump(lmodel, 'filename')
```
