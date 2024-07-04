from __future__ import division, print_function
from builtins import input



import pyfftw   # See https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np
from numpy import asarray
import os, glob, random

from sporco.admm import cbpdn, ccmod
from sporco.dictlrn import bpdndl
from sporco import util, signal, array, plot, cuda, fft
import sporco.metric as sm
import sporco_cuda.cbpdn as cucbpdn
plot.config_notebook_plotting()
sys_pipes = util.notebook_system_output()

#Loading training images
exim = util.ExampleImages(scaled=True, gray=False)



def loading_images(arr, num, directory):
    j = 0
    for images in os.listdir(directory + str(num)):
        if (images.endswith(".png")) and j <1000:
            arr.append(exim.image(directory + str(num) + '/' + images))
            j += 1
    return arr
    
def loading_random_images(arr, num, directory):
    all_pngs =  glob.glob(directory + str(num) + "/*.png")
    for j in range(1000):
        rand_png = random.choice(all_pngs)
        arr.append(exim.image(rand_png))
    return arr

def dict_learn(lmbda, dict_size): 


#Set parameters: block size whxwh, dict_size, lambda

    wh = 17
    blksz = (wh,wh)

    save_path = './images/MNIST/DL/' + str(wh) + 'x' + str(wh) + 'x' + str(dict_size) + '/lmbda_' + str(lmbda)
    folder_dir = './datasets/mnist_png/train/'


    if not os.path.exists('./images/MNIST/DL/' + str(wh) + 'x' + str(wh) + 'x' + str(dict_size)):
        os.makedirs('./images/MNIST/DL/' + str(wh) + 'x' + str(wh) + 'x' + str(dict_size))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    i = 0
    imgs = []

    for numbers in range(10):
        loading_random_images(imgs, numbers, folder_dir)
        print('Loading number ' + str(numbers) + ' complete')
   
   
    S = np.dstack(imgs)     
#Extract all image blocks, reshape.

    S  = array.extract_blocks(S, blksz)
    S = np.reshape(S, (np.prod(S.shape[0:2]), S.shape[2]))

#Construct initial dictionary.

    np.random.seed(12345)
    D0 = np.random.randn(S.shape[0], dict_size)

#Set regularization parameter and options for dictionary learning solver.


    opt = bpdndl.BPDNDictLearn.Options({'Verbose': True, 'MaxMainIter': 200,
                      'BPDN': {'rho': 10.0*lmbda + 0.1},
                      'CMOD': {'rho': S.shape[1] / 1e3}})

#Create solver object and solve.

    d = bpdndl.BPDNDictLearn(D0, S, lmbda, opt)
    D1 = d.solve()
    print("BPDNDictLearn solve time: %.2fs" % d.timer.elapsed('solve'))

    np.save('./dict_upload/dict_dl_lam_' + str(lmbda) + '_' + str(wh) + 'x' + str(wh) + 'x' + str(dict_size) + '.npy', D1)

#Display initial and final dictionaries.

    D1 = d.getdict().reshape((wh,wh, D0.shape[1]))


    X = d.getcoef()

    print('X: ' + str(X.shape))

#Save dictionary 


    D0 = D0.reshape(wh,wh, D0.shape[-1])


    fig = plot.figure(figsize=(14, 7))
    plot.imview(util.tiledict(D1), title='Słownik D', fig=fig)
    plot.savefig(save_path + '/dict_'+ str(lmbda) + '.png')

