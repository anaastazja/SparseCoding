from mnist_sparsecoding import mnist_sparsecoding
import time

lambdas = [1]
dict_sizes = [15]
start_full = time.time()


for d in range(len(dict_sizes)):
    print('******************************** DICT SIZE ' + str(dict_sizes[d]) + ' ****************************************')

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
end_full = time.time()
print('Execution time :' + str((end_full - start_full)/3600) + ' h')
