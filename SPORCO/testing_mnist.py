from mnist_sparsecoding import mnist_sparsecoding
import time

lambdas = [1.5]

start_tot = time.time()

for l in range(len(lambdas)):
    start = time.time()
    print('******************************************  LAMBDA ' + str(lambdas[l]) + '  ************************************************')
    for n in range(10):
        print('***************************  NUM  ' + str(n) + '  ***********************************')
        mnist_sparsecoding(lambdas[l], n)
    end = time.time()
    print('Time for one lambda: ' + str(end-start) + ' s')
        
        
end_tot = time.time()

print('Total time: ' + str((end_tot - start_tot)/60) + ' min')
