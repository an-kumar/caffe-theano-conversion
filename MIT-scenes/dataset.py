import theano
import time
import numpy as np
import theano.tensor as T
import os
import sys
from skimage import io, transform, util


'''
STRATEGY FOR GPU/CPU STUFF:

have: solver_batch_index, solver_batch_size
gpu_batch_index, gpu_batch_size
cpu_batch_index, cpu_batch_size.

then compute solver_batches_per_cpu_batch

given a solver_batch_index, we compute the gpu_batch_index and the cpu_batch_index as follows:

gpu_batch_index = solver_batch_index // solver_batches_per_gpu_batch
so, if this changes, then we must re-load the GPU from the CPU. Now, what happens if we're at the end of the CPU?

well, then the following will switch over:
cpu_batch_index = gpu_batch_index // gpu_batches_per_cpu_batch
and then we must re-load the CPU. now, note that gpu_batch_index INTO the cpu should be 0.

hence, real_gpu_batch_index is gpu_batch_index MODULO gpu_batches_per_cpu_batch.

if necessary, we update the necessary stuff.
'''

tensor5 = T.TensorType(theano.config.floatX, (False,)*5)
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
    elif len(shape) == 5:
        return tensor5



class ImageDirectoryDataset(object):
    '''
    for now, extremely simple. todo: decompose to BaseDataset
    '''
    def __init__(self, maindir_path, num_CPU_store, num_GPU_store, window_shape=(3,64,64), window_step=32):
        '''
        ASSUMPTIONS:

        maindir/
            train/
                class1
                class2
                ...
            test/
                class1
                class2
                ...
        '''
        #self.check_dir(maindir_path)
        self.window_shape= window_shape
        self.window_step = window_step

        # data housekeeping
        self.num_CPU_store = num_CPU_store
        self.num_GPU_store = num_GPU_store
        self.l2i, self.i2l, self.all_train_files, self.all_test_files = self.init_dir(maindir_path)

        self.gpu_batches_per_cpu_batch = int(np.ceil(num_CPU_store/float(num_GPU_store)))

        self.window_shape= window_shape
        self.window_step = window_step
        self.init_train()

        self.X_batch_var = self.get_X_batch_var()
        self.y_batch_var = self.get_y_batch_var()

        self.curr_gpu_batch = 0
        self.curr_cpu_batch = 0




    def get_X_batch_var(self):
        return tensortype_from_shape(self.CPU_X_train.shape)()
    def get_y_batch_var(self):
        return tensortype_from_shape(self.CPU_y_train.shape, intg=True)()


    def init_train(self):
        self.CPU_X_train, self.CPU_y_train = self.load_cpu(0,self.num_CPU_store)
        self.GPU_X_train, self.GPU_y_train = self.load_gpu(0, self.num_GPU_store)

    def load_cpu(self, batch_index, batch_size, mode='train'):
        '''
        loads a batch of data into the cpu

        batch_index is the index into the disk!
        '''
                tick = time.time()
        if mode =='train':
            files = self.all_train_files[batch_index*batch_size: (batch_index+1)*batch_size]
        elif mode == 'test':
            files = self.all_test_files[batch_index*batch_size:(batch_index+1)*batch_size]
        else:
            raise Exception ("mode %s not valid for load_cpu" % mode)

        # todo: make the following general, for decomposition
        X_batch = []
        y_batch = []
        for filename in files:
            X, y = self.process_single_file(filename) # this is because process_single_file will augment
            X_batch.append(X)
            y_batch.append(y)

        tock = time.time()
        print "time taken:%s" % str(tock - tick)
        return np.array(X_batch), np.array(y_batch)

    def process_single_file(self, filename):
        '''
        processes a single image file
        '''
        img = io.imread(filename)
        img = img.transpose(2,0,1)
                img = transform.resize(img, (3,227,227))
        label = filename.split('/')[-2] # HACKY!
        y = self.l2i[label]

        # view is 1 x a x b x window shape.
        view = util.view_as_windows(img, window_shape=self.window_shape, step=self.window_step)
        # we want this to be num_windows x windows shape, so:
        batches = view.reshape(np.prod(view.shape[:-3]), *view.shape[3:])
        new_batches = []
        for batch in batches:
            new_batches.append(transform.resize(batch,(3,227,227)))
        return np.array(new_batches), y

    def load_gpu(self, batch_index, batch_size):
        '''
        loads a batch of data into the cpu

        batch_index is the index into the CPU batch!

        '''
        X_gpu = theano.shared(self.CPU_X_train[batch_index*batch_size:(batch_index+1)*batch_size].astype(theano.config.floatX)) 
        # T.cast for decomposibility???
        y_gpu = T.cast(theano.shared(self.CPU_y_train[batch_index*batch_size:(batch_index+1)*batch_size].astype(theano.config.floatX)), 'int32')
                return X_gpu, y_gpu




    def check_dir(maindir_path):
        sub_dirs = os.listdir(maindir_path)
        if set(sub_dirs) != set(['train','test']):
            raise Exception('dir needs to be set up folders of train, test')


    def init_dir(self, maindir_path):
        '''
        builds up the label to index, index to label mappings. also, stores the train, test.
        '''
        l2i, i2l = self.build_mappings(maindir_path)
        train_dir = os.path.join(maindir_path,'train')
        test_dir = os.path.join(maindir_path, 'test')

        all_train_files = []
        for label in os.listdir(train_dir):
            label_dir = os.path.join(train_dir, label)
            all_train_files += [os.path.join(label_dir, x) for x in os.listdir(label_dir)]

        all_test_files = []
        for label in os.listdir(test_dir):
            label_dir = os.path.join(test_dir, label)
            all_test_files += [os.path.join(label_dir, x) for x in os.listdir(label_dir)]

        return l2i, i2l, all_train_files, all_test_files


    def build_mappings(self, maindir_path):
        # store classes from train
        train_dir = os.path.join(maindir_path, 'train')
        # class list is os.listdir(train_dir)
        i2l = os.listdir(train_dir)
        l2i = {i2l[i]:i for i in range(len(i2l))}
        return l2i, i2l

    def deal_with_batch(self, batch_index, batch_size, mode='train'):
        '''
        here, batch is in fact NOT a theano variables
        '''
        solver_batches_per_gpu_batch = self.num_GPU_store / batch_size
        # this is the global gpu batch index
        gpu_batch_index = batch_index / solver_batches_per_gpu
        if gpu_batch_index != self.curr_gpu_batch:
            # now we need to load next batch of the gpu
            # ===== [] ####
            # this is the global cpu batch
            cpu_batch_index = gpu_batch_index / self.gpu_batches_per_cpu_batch
            if cpu_batch_index != self.curr_cpu_batch:
                # now we need to load the next batch of the cpu
                self.load_cpu(cpu_batch_index, self.num_CPU_store, mode='train')
                self.curr_cpu_batch = cpu_batch_index
            # ==== [] ###
            # ok, now we have a new cpu batch. now we want to set the gpu values.
            real_gpu_batch = gpu_batch_index % self.gpu_batches_per_cpu_batch
            self.set_GPU(real_gpu_batch)

    def set_gpu(self, real_gpu_batch):
        self.GPU_X_train.set_value(self.CPU_X_train[real_gpu_batch*self.num_GPU_store:(real_gpu_batch+1)*self.num_GPU_store])
        self.GPU_y_train.set_value(self.CPU_y_train[real_gpu_batch*self.num_GPU_store:(real_gpu_batch+1)*self.num_GPU_store])




    def train_givens(self, batch_index, batch_size):
        '''
        batch_index is a theano_variable.
        '''
        # compute the gpu batch index
        # these will all be theano variables
        solver_batches_per_gpu_batch = T.cast(T.int_div(self.num_GPU_store,batch_size), 'int32')
        real_batch_index = T.cast(T.mod(batch_index, solver_batches_per_gpu_batch), 'int32')

        givens = {self.X_batch_var[i]:self.X_trains[i][real_batch_index*batch_size:(real_batch_index+1)*batch_size] for i in range(len(self.X_trains)) }
        givens[self.y_batch_var] = self.y_train[real_batch_index*batch_size:(real_batch_index+1)*batch_size]
        return givens

    def test_givens(self):
        givens = {self.X_batch_var[i]:self.X_tests[i] for i in range(len(self.X_tests))}
        givens[self.y_batch_var] = self.y_test
        return givens       

if __name__ == '__main__':
    ds = ImageDirectoryDataset('../../proj/Images', 100, 25)

