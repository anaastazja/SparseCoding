from __future__ import division
from __future__ import print_function
from builtins import input

import pyfftw   # See https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np
from numpy import asarray
import os, random, glob, csv, math
from matplotlib import pyplot as plt
import matplotlib as mpl

import sporco.metric as sm
from sporco.admm import cbpdn, ccmod
from sporco.dictlrn import dictlrn, cbpdndl
from sporco import cnvrep, util, signal, plot, cuda, fft
import sporco_cuda.cbpdn as cucbpdn
from sporco.pgm.backtrack import BacktrackStandard

plot.config_notebook_plotting()
sys_pipes = util.notebook_system_output()


def convsparsecoding(lmbda, num, dict_size):
    wh = 17
    sqrt_dict_size = int(np.sqrt(dict_size))   
    
#Loading training images
    exim = util.ExampleImages(scaled=True, gray=False)

#CSV headers
    headers = ['number', 'absolute_error', 'mean_error']
    headers_sparsity = ['number', 'dict_size', 'lmbda', 'sparsity_count', 'percent']

#Folders' paths
    
    folder_test_dir = './datasets/mnist_png/valid/'
    save_path = './images/MNIST/CDL/' + str(wh) + 'x' + str(wh) + 'x' + str(dict_size) + '/lmbda_' + str(lmbda)


#Create folder if not exist


    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(save_path + '/data_lmbda_' + str(lmbda) + '.csv'):
        with open(save_path + '/data_lmbda_' + str(lmbda) + '.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
    if not os.path.exists('./images/MNIST/CDL/' + str(wh) + 'x' + str(wh) + '.csv'):
        with open('./images/MNIST/CDL/' + str(wh) + 'x' + str(wh) + '.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers_sparsity)

    D1 = np.load('./convdict_upload/dict_cdl_lam_' + str(lmbda) + '_' + str(wh) + 'x' + str(wh) + 'x' + str(dict_size) + '.npy')

#TEST img


#Set admm.cbpdn.ConvBPDN solver options.

    opt = cbpdn.ConvBPDN.Options({'Verbose': False, 'MaxMainIter': 100,
		                      'RelStopTol': 5e-3, 'AuxVarObj': False})

#Highpass filter example image.
    npdd = 16 
#Number of samples to pad at image boundaries
    fltlmbdd = 10 

#Initialise and run CSC solver.
    i = 0
    all_pngs =  glob.glob(folder_test_dir + str(num) + "/*.png")
    if cuda.device_count() > 0:
        print('%s GPU found: running CUDA solver' % cuda.device_name())
        for i in range(500):
#Load image
            rand_png = random.choice(all_pngs)
            img = exim.image(rand_png)  
	    #Regularization parameter controlling lowpass filtering
            sll, shh = signal.tikhonov_filter(img, fltlmbdd, npdd)
            tm = util.Timer() 
	    #Reconstruct image from sparse representation.
            with sys_pipes(), util.ContextTimer(tm):
                X = cuda.cbpdn(D1, shh, lmbda, opt)
                shr = np.sum(fft.fftconv(D1, X, axes=(0, 1)), axis=2)
                imgr = sll + shr
            t = tm.elapsed()
            abs_error = sm.mae(img, imgr)
            mean_error = sm.mse(img, imgr)
    
            data = [num, abs_error, mean_error]
    
            with open(save_path + '/data_lmbda_' + str(lmbda) + '.csv', 'a', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data)
    else:
        print('GPU not found: running Python solver')
        for i in range(5):
        #Load image
            rand_png = random.choice(all_pngs)
            img = exim.image(rand_png)  
#Regularization parameter controlling lowpass filtering  
            sll, shh = signal.tikhonov_filter(img, fltlmbdd, npdd)
#Reconstruct image from sparse representation.
            c = cbpdn.ConvBPDN(D1, shh, lmbda, opt, dimK=0)
            X = c.solve()
            t = c.timer.elapsed('solve')
            print('Solve time: %.2f s' % t)
            shr = np.sum(fft.fftconv(D1, X, axes=(0, 1)), axis=2)
            imgr = sll + shr
            print("Reconstruction PSNR: %.2fdB\n" % sm.psnr(img, imgr))
            abs_error = sm.mae(img, imgr)
            mean_error = sm.mse(img, imgr)
    
            data = [num, abs_error, mean_error]
    
            with open(save_path + '/data_lmbda_' + str(lmbda) + '.csv', 'a', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data)

    count_sparsity = 0
    pixels = 0
    for v in np.nditer(X):
        pixels += 1
        if abs(v) < 1e-6:
            count_sparsity += 1

    percent = count_sparsity/pixels*100
    data_sparsity = [num, dict_size, lmbda, count_sparsity, percent]
    
    with open('./images/MNIST/CDL/' + str(wh) + 'x' + str(wh) + '.csv', 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data_sparsity)

	
    if X.max() >= abs(X.min()):
        val = X.max()
    else: 
        val = abs(X.min())
    n = 0	
    fig, axes = plt.subplots(ncols=math.ceil(dict_size/sqrt_dict_size), nrows=sqrt_dict_size, sharex=True, sharey=True)	

    for ax in axes.flat:
        if n < dict_size:
            im = ax.imshow(X[:, :, n], cmap = 'seismic', vmin = -1*math.ceil(val), vmax = math.ceil(val))
            ax.axes.yaxis.set_ticklabels([])
            ax.axes.xaxis.set_ticklabels([])
            n += 1
    cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
    plt.colorbar(im, cax=cax, **kw)
    plt.suptitle('Mapa współczynników dla cyfry \"' + str(num) + '\"')
    plt.savefig(save_path + '/X_cdl_' + str(num) + '_lmbda_' + str(lmbda) + '.png')	    


#Display original and reconstructed images.

    fig = plot.figure(figsize=(14, 7))
    plot.subplot(1, 2, 1)
    plot.imview(img, title='Oryginalny', fig=fig)
    plot.subplot(1, 2, 2)
    plot.imview(imgr, title='Rekonstrukcja', fig=fig)
    plot.savefig(save_path +'/econstruct_dl_' + str(num) + '_lmbda_' + str(lmbda) + '.png')
    plot.close()
