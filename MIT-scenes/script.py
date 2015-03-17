import sys
sys.path.insert(0,'../')
import dataset

ds = dataset.ImageDirectoryDataset('../../proj/Images', 250, 1) 
print "...train..."
ds.dump_to_pickles(mode='train')
print "...test..."
ds.dump_to_pickles(mode='test')
