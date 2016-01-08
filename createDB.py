import numpy as np
import lmdb
import caffe
from PIL import Image
import os

imgPath = './forLabeling'
N = 13233

# Let's pretend this is interesting data
X = np.zeros((N, 3, 250, 250), dtype=np.uint8)
# y = np.zeros(N, dtype=np.int64)

# We need to prepare the database for the size. We'll set it 10 times
# greater than what we theoretically need. There is little drawback to
# setting this too big. If you still run into problem after raising
# this, you might want to try saving fewer entries in a single
# transaction.

map_size = X.nbytes * 2

env = lmdb.open('alllfw_lmdb', map_size=map_size)

os.chdir(imgPath)

with env.begin(write=True, buffers=True) as txn:
    # txn is a Transaction object
    for i in range(N):
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = X.shape[1]
        datum.height = X.shape[2]
        datum.width = X.shape[3]
        
        imgIndex = i + 1
        datum.data = Image.open('%d.jpg' % (imgIndex)).tobytes()  # or .tostring() if numpy < 1.9
        # datum.label = int(y[i])
        str_id = '{:08}'.format(imgIndex)

        # The encode is only essential in Python 3
        txn.put(str_id.encode('ascii'), datum.SerializeToString())
