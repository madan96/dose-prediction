"""
Custom Dataset Implementation
"""

import glob
from options import Options
from six.moves import cPickle
import numpy as np
import os
from skimage import io
from scipy.misc import imsave

from matplotlib import pyplot as plt


class Dataset(object):
    def __init__(self, opts):
        self.opts = opts

        imgs_path = os.path.join(self.opts.root_dir, self.opts.dataset_dir)
        files = glob.glob(imgs_path + "/*.raw")
        self.images = np.empty((len(files), 4, 167, 256, 256), dtype="uint16")
        img_list = []
        for f in files:
            img = np.fromfile(f, dtype="uint16", sep="")
            img = img[36:] # Remove initial data
            img = np.reshape(img, (4, 167, 256, 256))
            img_list.append(img)
        self.images = np.asarray(img_list)
        self.images = self.images.astype('float')

        self.ptv = self.images[:,1,[60, 61, 62],:,:] / 255.0
        self.slice = self.images[:,0,[60, 61, 62],:,:]
        self.combined = self.images[:,2,[60, 61, 62],:,:] 
        self.oct = np.concatenate((self.slice, self.combined), axis=1) / 255.0
        self.dose = self.images[:,3,[60, 61, 62],:,:] / 255.0

    def __len__(self):
        return self.images.shape[0]

    def load_batch(self, start_idx, end_idx):
        return self.ptv[start_idx:end_idx], self.oct[start_idx:end_idx], self.dose[start_idx:end_idx]
