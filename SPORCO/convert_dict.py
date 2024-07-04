from __future__ import division, print_function
from builtins import input



import pyfftw   # See https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np
from numpy import asarray
import os

from sporco.admm import cbpdn, ccmod
from sporco.dictlrn import bpdndl
from sporco import util, signal, array, plot, cuda, fft
import sporco.metric as sm
import sporco_cuda.cbpdn as cucbpdn
plot.config_notebook_plotting()
sys_pipes = util.notebook_system_output()

lmbda = 1e-5

D = np.load('/home/anastazja/sporco-cuda/dict_upload/dict_dl_lam1e-057x7x25.npy')



D1 = D.reshape((7,7, D.shape[1]))

fig = plot.figure(figsize=(14, 7))
plot.imview(util.tiledict(D1), title='Słownik D', fig=fig)
plot.savefig('./images/MNIST/DL/7x7x25/dictlearn_lmbd' + str(lmbda) + '.png')
fig.show()
