from convdictlearn import convdictlearn
from mnist_convsparsecoding import convsparsecoding
import time
from numba import cuda 

lambdas = [0.5, 1, 1.5, 2]
dict_sizes = [7]
start_full = time.time()


for d in range(len(dict_sizes)):

    print('*********************************** DICT SIZE ' + str(dict_sizes[d]) + ' **************************************************')
    start_tot_test = time.time()
    for l in range(len(lambdas)):
        start_test = time.time()
        print('******************************************  LAMBDA ' + str(lambdas[l]) + '  ************************************************')
        for n in range(10):
            start = time.time()
            print('***************************  NUM  ' + str(n) + '  ***********************************')
            convsparsecoding(lambdas[l], n, dict_sizes[d])
            end = time.time()
            print('Done in: ' + str(end-start) + ' s')
            device = cuda.get_current_device()
            device.reset()
        end_test = time.time()
        print('Time for one lambda: ' + str((end_test - start_test)/60) + ' min')
        
        
    end_tot_test = time.time()
    print('Total time reconstructing: ' + str((end_tot_test - start_tot_test)/60) + ' min')
    
end_full = time.time()

print('Total execution time: ' + str((end_full-start_full)/60) + ' min')
