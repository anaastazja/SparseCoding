from __future__ import division
from __future__ import print_function
from builtins import input

import pyfftw   # See https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np
from numpy import asarray
import os, random, glob

import sporco.metric as sm
from sporco.admm import cbpdn, ccmod
from sporco.dictlrn import cbpdndl
from sporco import cnvrep, util, signal, plot, cuda, fft
from sporco.pgm.backtrack import BacktrackStandard

plot.config_notebook_plotting()
sys_pipes = util.notebook_system_output()
exim = util.ExampleImages(scaled=True, gray=False)

def loading_images(arr, num, directory):
    j = 0
    for images in os.listdir(directory + str(num)):
        if (images.endswith(".png")) and j<1000:
            arr.append(exim.image(directory + str(num) + '/' + images))
            j += 1
    return arr

def loading_random_images(arr, num, directory):
    all_pngs =  glob.glob(directory + str(num) + "/*.png")
    for j in range(1000):
        rand_png = random.choice(all_pngs)
        arr.append(exim.image(rand_png))
    return arr

#Set parameters: block size whxwh, dict_size, lambda

def convdictlearn(lmbda, dict_size):

    wh = 17
    L_du = 50
#Paths to folders

    folder_dir = './datasets/mnist_png/train/'
    save_path = './images/MNIST/CDL/' + str(wh) + 'x' + str(wh) + 'x' + str(dict_size) + '/lmbda_' + str(lmbda)

    if not os.path.exists('./images/MNIST/CDL/' + str(wh) + 'x' + str(wh) + 'x' + str(dict_size)):
        os.makedirs('./images/MNIST/CDL/' + str(wh) + 'x' + str(wh) + 'x' + str(dict_size))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    imgs = []


    for numbers in range(10):
        loading_random_images(imgs, numbers, folder_dir)
        print('Loading number ' + str(numbers) + ' complete')

    S = np.dstack(imgs)
#Construct initial dictionary.

    np.random.seed(12345)
    D0 = np.random.randn(wh, wh, dict_size)

#Create DictLearn object and solve.

    opt = cbpdndl.ConvBPDNDictLearn.Options({'Verbose': True, 'MaxMainIter': 200,
			    'AccurateDFid': True,
                            'CBPDN': {'rho': 50.0*lmbda + 0.5}, 
                            'CCMOD': {'Backtrack': BacktrackStandard(), 'L': L_du}}, 
                            xmethod = 'admm', 
                            dmethod = 'pgm')
                            
    d = cbpdndl.ConvBPDNDictLearn(D0, S, lmbda, opt)

    D1 = d.solve()
    print("ConvBPDNDictLearn solve time: %.2fs" % d.timer.elapsed('solve'))

#Display Dictionary

    D1 = D1.squeeze()

    fig = plot.figure(figsize=(14, 7))
    plot.imview(util.tiledict(D1), title='Słownik D', fig=fig)
    plot.savefig(save_path + '/dict_' + str(lmbda) + '.png')
    plot.close()
#Save Dictionary
    np.save('./convdict_upload/dict_cdl_lam_' + str(lmbda) + '_' + str(wh) + 'x' + str(wh) + 'x' + str(dict_size) + '.npy', D1)
