from convdictlearn import convdictlearn
from mnist_convsparsecoding import convsparsecoding
import time
from numba import cuda 

lambdas = [0.1, 0.5, 1, 1.5, 2]
dict_sizes = [25]
start_full = time.time()


for d in range(len(dict_sizes)):
    print('******************************** DICT SIZE ' + str(dict_sizes[d]) + ' ****************************************')    
    start_tot = time.time()
    for i in range(len(lambdas)):
        start = time.time()
        print('******************************** LAMBDA ' + str(lambdas[i]) + ' ****************************************')
        convdictlearn(lambdas[i], dict_sizes[d])
        end = time.time()
        print('Time: ' + str(end - start) + ' s')
    
    end_tot = time.time()
    print('Total time learning dictionaries: ' + str((end_tot - start_tot)/3600) + ' h')


    start_tot_test = time.time()

    for l in range(len(lambdas)):
        start_test = time.time()
        print('******************************************  LAMBDA ' + str(lambdas[l]) + '  ************************************************')
        for n in range(10):
            print('***************************  NUM  ' + str(n) + '  ***********************************')
            start2 = time.time()
            convsparsecoding(lambdas[l], n, dict_sizes[d])
            end2 = time.time()
            print('Done in: ' + str(end2-start2) + ' s')
            device = cuda.get_current_device()
            device.reset()
        end_test = time.time()
        print('Time for one lambda: ' + str((end_test - start_test)/60) + ' min')
        
        
    end_tot_test = time.time()

    print('Total time reconstructing: ' + str((end_tot_test - start_tot_test)/60) + ' min')    
    print('Total time learning dictionaries: ' + str((end_tot - start_tot)/3600) + ' h')
end_full = time.time()
print('Execution time :' + str((end_full - start_full)/3600) + ' h')
