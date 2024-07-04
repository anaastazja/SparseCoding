from convdictlearn import convdictlearn
import time

lambdas = [0.1, 0.5, 1, 1.5, 2]
dict_sizes = [10, 11, 14, 15, 16, 18, 20, 25]
start_full = time.time()


for d in range(len(dict_sizes)):
    print('******************************** DICT SIZE ' + str(dict_sizes[d]) + ' ****************************************')    
    start_tot = time.time()
    for i in range(len(lambdas)):
        start = time.time()
        print('******************************** LAMBDA ' + str(lambdas[i]) + ' ****************************************')
        convdictlearn(lambdas[i], dict_sizes[d])
        end = time.time()
        print('Time: ' + str((end - start)/60) + ' min')
    
    end_tot = time.time()
    print('Total time learning dict_size ' + str(d) + ' dictionaries: ' + str((end_tot - start_tot)/3600) + ' h')
end_full = time.time()
print('Execution time :' + str((end_full - start_full)/3600) + ' h')
