import csv
import mono_layer

import numpy as np
import matplotlib.pyplot as plt

import sys
from datetime import datetime



def load_titanic(train_file_path, verbose = False):
    '''
    loads the special titanic dataset
    '''
    train_l = 0
    train_x = []
    train_y = []
    test_l = 0
    test_x = []
    test_y = []
    
    file = open(train_file_path)
    reader = csv.reader(file)
    next(reader, None)
    for row in reader:

        row_x = []
        
        # passenger class
        row_x.append(int(row[2]))

        #passenger sex
        if row[4] == 'male':
            row_x.append(1)
        elif row[4] == 'female':
            row_x.append(2)
        else:
            if verbose:
                print('error in reading SEX in row', row[0], '-' ,row [4], 'registered as 0')
            row_x.append(0)

        # age, sib, parch, fare
        try:
            row_x.append(int(row[5]))
        except :
            row_x.append(0)

        if row[6]:
            row_x.append(int(row[6]))
        else:
            row_x.append(0)
        
        if row[7]:
            row_x.append(int(row[7]))
        else:
            row_x.append(0)
        # row_x.append(int(row[9]))

        
        # port
        if 'S' in row[11]:
            row_x.append(1)
        elif 'C' in row[11]:
            row_x.append(2)
        elif 'Q' in row[11]:
            row_x.append(3)
        else:
            row_x.append(0)
            if verbose:
                print('error in reading PORT in row', row[0], ' - ' ,row [11], 'registered as 0')

        # assign a row value            
        if train_l < 500:
            train_l += 1
            train_x.append(row_x)
            if row[1]:
            	train_y.append(int(row[1]))
        else:
            test_l += 1
            test_x.append(row_x)
            if row[1]:
            	test_y.append(int(row[1]))
    file.close()
    
    X_train = np.array(train_x).T
    Y_train = np.array(train_y).reshape(1,train_l)
    X_test = np.array(test_x).T
    Y_test = np.array(test_y).reshape(1,test_l)

    if verbose:
        print()
        print ("train_set_x_flatten shape: " + str(X_train.shape))
        print ("train_set_y shape: " + str(Y_train.shape))
        print ("test_set_x_flatten shape: " + str(X_test.shape))
        print ("test_set_y shape: " + str(Y_test.shape))
        print()

    return(X_train, Y_train, X_test, Y_test)


train_set_x, train_set_y, test_set_x, test_set_y = load_titanic('titanic data/train.csv',verbose = False)


def single_hidden_layer_test(verbose = True):

    n_tests = 6
    tests = {}
    if verbose:
        print("\nstarting testing of",n_tests,"cases\n")
    num_iterations = 25000

    for i in range(n_tests):
        learning_rate = np.abs(round(np.random.rand()  * 0.4,3))
        hidden_layers = np.abs(int((np.random.randn() + 2)*3))
        if (verbose):
            print()
            print ("test #",i,"\tNum iterations = ", num_iterations,"\tLearning rate =", learning_rate,"\tHidden units",hidden_layers,"\n")
        tests[str(i)] = mono_layer.nn_model(train_set_x, train_set_y, test_set_x, test_set_y, 7, num_iterations, learning_rate, num_iterations/10,False)
        tests[str(i)]["hidden_layers"] = hidden_layers
        tests[str(i)]["learning_rate"] = learning_rate
        tests[str(i)]["num_iterations"] = num_iterations
    return tests

def graph_it(tests):
    plt.figure(figsize=(10.90, 8.00), dpi=150)
    for i in range(len(tests)):
        costs = tests[str(i)]["sample_costs"]
        iterations = range(0,tests[str(i)]["num_iterations"]+1,int(tests[str(i)]["num_iterations"]/10))

        plt.plot(iterations,costs,label = \
            "Test #" + str(i) + \
            " Learning Rate = " + str(tests[str(i)]["learning_rate"]) + \
            " Hidden Units = " + str(tests[str(i)]["hidden_layers"]) \
            )
        #plt.text(iterations[-1], costs[-1], 'test {i}'.format(i=i))
    plt.ylabel('cost')
    plt.legend(loc='upper right', frameon=True)
    plt.xlabel('iterations')
    plt.title("Evolution of costs")
    plt.grid(axis='y', linestyle='-')
    plt.savefig("figure 1.png", facecolor='w', edgecolor='k')
    plt.show()


def map_it(tests):
    exclude = ["W1","b1","W2","b2"]
    for t in tests:
        print("TEST",t)
        for data in tests[t]:
            if data not in exclude and data != "sample_costs":
                print (data,"\t",tests[t][data])
            elif data == "sample_costs":
                print ("final cost","\t",tests[t][data][-1])
        print()


log = open('log.txt', 'w')
sys.stdout = log

np.random.seed(1)
tests_results = single_hidden_layer_test(True)
map_it(tests_results)
graph_it(tests_results)

print("\nDONE fellow human \n")

log.close()