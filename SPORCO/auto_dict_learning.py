from dictlearn import dict_learn
from mnist_sparsecoding import mnist_sparsecoding
import time

lambdas = [1, 1.5]
dict_sizes = [7, 8, 9, 10, 11, 12, 14, 15, 16, 18, 20, 25]
start_full = time.time()

start_tot = time.time()

for d in range(len(dict_sizes)):
    print('******************************** DICT SIZE ' + str(dict_sizes[d]) + ' ****************************************')
    for i in range(len(lambdas)):
        start = time.time()
        print('******************************** LAMBDA ' + str(lambdas[i]) + ' ****************************************')
        dict_learn(lambdas[i], dict_sizes[d])
        end = time.time()
        print('Time: ' + str(end - start) + ' s')
    
    end_tot = time.time()
    print('Total time learning dictionaries: ' + str((end_tot - start_tot)/60) + ' min')


    start_tot_test = time.time()

    for l in range(len(lambdas)):
        start_test = time.time()
        print('******************************************  LAMBDA ' + str(lambdas[l]) + '  ************************************************')
        for n in range(10):
            print('***************************  NUM  ' + str(n) + '  ***********************************')
            mnist_sparsecoding(lambdas[l], n, dict_sizes[d])
        end_test = time.time()
        print('Time for one lambda: ' + str((end_test - start_test)/60) + ' min')
        
        
    end_tot_test = time.time()

    print('Total time reconstructing: ' + str((end_tot_test - start_tot_test)/60) + ' min')    
    print('Total time learning dictionaries: ' + str((end_tot - start_tot)/3660) + ' h')
end_full = time.time()
print('Execution time :' + str((end_full - start_full)/3600) + ' h')
