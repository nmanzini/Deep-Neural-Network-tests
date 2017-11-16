import csv

import logistic_regression 
import mono_layer
import dnn

import numpy as np
import matplotlib.pyplot as plt


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


train_set_x, train_set_y, test_set_x, test_set_y = load_titanic('titanic data/train.csv',True)

# d = logistic_regression.model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 40000, learning_rate = 0.002, print_cost = 5000)

num_iterations = 1000000
learning_rate = 0.01


'''
d = mono_layer.nn_model(train_set_x, train_set_y, test_set_x, test_set_y, 7, num_iterations, learning_rate, print_cost=True)
costs_1 = d["costs"]
d = logistic_regression.model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations, learning_rate, print_cost= 1000)
costs_0 = d["costs"]
'''

layers_dims = [6,3,1] #  5-layer model
parameters = dnn.L_layer_model(train_set_x, train_set_y, layers_dims, num_iterations = 100000, print_cost = True, learning_rate = 0.002)
# Plot learning curve (with costs)

pred_train = dnn.predict(parameters, train_set_x)
pred_test = dnn.predict(parameters, test_set_x)


print("train accuracy: {} %".format(100 - np.mean(np.abs(pred_train - train_set_y)) * 100))
print("train accuracy: {} %".format(100 - np.mean(np.abs(pred_test - test_set_y)) * 100))

print (parameters)

'''
plt.plot(range(num_iterations),costs_0)
plt.plot(range(num_iterations),costs_1,)
plt.ylabel('cost')
plt.xlabel('iterations')
plt.title("Evolution of costs")
plt.show()

print("final costs_0", costs_0[-1])
print("final costs_1", costs_1[-1])

print("\nDONE fellow human \n")
'''