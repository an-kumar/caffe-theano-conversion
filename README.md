# caffe_theano_conversion

usage:
```
>>> from conversion import convert
>>> from models import *
>>> lmodel = convert('/path/to/deploy.prototxt', '/path/to/pretrained_model.caffemodel')
>>> dump(lmodel, 'filename')```