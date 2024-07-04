from __future__ import division, print_function
from builtins import input

import pyfftw   # See https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np
from numpy import asarray
import os, random, glob
from matplotlib import pyplot as plt
import matplotlib as mpl
import math
import csv

from sporco.admm import bpdn
from sporco import util, signal, array, plot, cuda, fft
import sporco.metric as sm
import sporco_cuda.cbpdn as cucbpdn
from sporco.pgm.backtrack import BacktrackStandard



plot.config_notebook_plotting()
sys_pipes = util.notebook_system_output()


def mnist_sparsecoding(lmbda, num, dict_size):

#Loading training images
    exim = util.ExampleImages(scaled=True, gray=False)

#Set block size whxwh and dict_size

    wh = 13
    sqrt_dict_size = int(np.sqrt(dict_size))
    blksz = (wh, wh)
    rsh_size = math.ceil(wh/2) - 1


#Folders' paths

    headers = ['number', 'absolute_error', 'mean_error']
    headers_sparsity = ['number', 'dict_size', 'lmbda', 'sparsity_count', 'percent']
    
    folder_test_dir = './datasets/mnist_png/valid/'
    save_path = './images/MNIST/DL/' + str(wh) + 'x' + str(wh) + 'x' + str(dict_size) + '/lmbda_' + str(lmbda)



#Create folder if not exist



    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(save_path + '/data_lmbda_' + str(lmbda) + '.csv'):
        with open(save_path + '/data_lmbda_' + str(lmbda) + '.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
    if not os.path.exists('./images/MNIST/DL/' + str(wh) + 'x' + str(wh) + '.csv'):
        with open('./images/MNIST/DL/' + str(wh) + 'x' + str(wh) + '.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers_sparsity)


#Load Dictionary

    D = np.load('./dict_upload/dict_dl_lam_' + str(lmbda) + '_' + str(wh) + 'x' + str(wh) + 'x' + str(dict_size) + '.npy')
	
    D_check = D
    print(D_check.shape)
#TEST img

#Set admm.cbpdn.ConvBPDN solver options.

    opt = bpdn.BPDN.Options({'Verbose': False, 'MaxMainIter': 250,
                         'RelStopTol': 3e-3, 'AuxVarObj': False,
                         'AutoRho': {'Enabled': False}, 'rho':
                         1e1*lmbda})

    i = 0
    all_pngs =  glob.glob(folder_test_dir + str(num) + "/*.png")
    for i in range(1):
        rand_png = random.choice(all_pngs)
        img = exim.image(rand_png)
#Convert image into blocks 
        blocks = array.extract_blocks(img, blksz)
        blocks = blocks.reshape(np.product(blksz), -1)
                        
        b = bpdn.BPDN(D, blocks, lmbda, opt)
        x1 = b.solve()

        print(str(i) + " ADMM BPDN solve time: %.2fs" % b.timer.elapsed('solve'))
       
        imgr = np.dot(D, x1).reshape(blksz + (-1,))
        imgr = array.combine_blocks(imgr, img.shape)
        
        abs_error = sm.mae(img, imgr)
        mean_error = sm.mse(img, imgr)
    
        data = [num, abs_error, mean_error]
    
        with open(save_path + '/data_lmbda_' + str(lmbda) + '.csv', 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)
 

		    
#Display coefficient maps
    print(np.array_equal(D_check, D))
    X1 = x1.reshape((dict_size,28 - 2*rsh_size,28 - 2*rsh_size))

    count_sparsity = 0
    pixels = 0
    for v in np.nditer(X1):
        pixels += 1
        if abs(v) < 1e-6:
            count_sparsity += 1
    print(pixels)
    percent = count_sparsity/pixels*100
    data_sparsity = [num, dict_size, lmbda, count_sparsity, percent]
    
    with open('./images/MNIST/DL/' + str(wh) + 'x' + str(wh) + '.csv', 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data_sparsity)
    
    if X1.max() >= abs(X1.min()):
        val = X1.max()
    else: 
        val = abs(X1.min())
    n = 0
    fig, axes = plt.subplots(nrows=math.ceil(dict_size/sqrt_dict_size), ncols=sqrt_dict_size, sharex=True, sharey=True)
    
    for ax in axes.flat:
        if n < dict_size:
            im = ax.imshow(X1[n, :, :], cmap = 'seismic', vmin = -1*math.ceil(val), vmax = math.ceil(val))
            ax.axes.yaxis.set_ticklabels([])
            ax.axes.xaxis.set_ticklabels([])
            n += 1
    cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
    plt.colorbar(im, cax=cax, **kw)
    plt.suptitle('Mapa współczynników dla cyfry \"' + str(num) + '\"')
    plt.savefig(save_path + '/X_dl_' + str(num) + '_lmbda_' + str(lmbda) + '.png')
#Display original and reconstructed images.

    fig = plot.figure(figsize=(14, 7))
    plot.subplot(1, 2, 1)
    plot.imview(img, title='Original', fig=fig)
    plot.subplot(1, 2, 2)
    plot.imview(imgr, title='Reconstructed', fig=fig)
    plot.savefig(save_path +'/econstruct_dl_' + str(num) + '_lmbda_' + str(lmbda) + '.png') 
    plot.close()  
