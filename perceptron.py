import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets

def prediction(data, weights, bias):
    '''
    This routine is used to predict the target using the weights and bias.
    Example:
        data = [1,2,3,4]
        weights = [0,0,1,1]
        bias = 1
        target = prediction(data, weights, bias)
    '''
    result = 0
    for index in range(len(data)):
        result += (weights[index] * data[index])
    result += bias
    if result >= 0:
        target = 1
    else:
        target = 0
    return target

def accuracy(dataset, weights, bias, target):
    '''
    This routine returns the percentage of accuracy for the dataset and targets given
    Example:
        dataset = [[1,2,3,4],[1,2,3,4]]
        weights = [0,0,1,1]
        bias = 1
        target = [0,1]
        accy = accuracy(dataset, weights, bias, target)
    '''
    success = 0
    num_of_elements = len(dataset)

    for element in range(num_of_elements):
        guess = prediction(dataset[element], weights, bias)
        if guess == target[element]:
            success += 1
    
    accy = success/num_of_elements
    return accy

def training(dataset, target, bias, weight, learning_factor, learn_bias=True):
    '''
    This routine adjust the weights and bias according to the given dataset and target. 
    The bias can be adjusted or explicit defined, depends on the flag "learn_bias".
    Example:
        dataset = [[1,2,3,4],[1,2,3,4]]
        weight = [0,0,1,1]
        bias = 1
        target = [0,1]
        learning_factor = 2
        learn_bias = True
        final_bias,final_weight = training(dataset, target, bias, weight, learning_factor, learn_bias=True)
    '''
    final_bias = bias
    final_weight = np.copy(weight)
    #print("===============================")
    #print("       Start of Training")
    #print("===============================")
    #print()


    for attempts in range(25):
        #print("-----------------------")
        #print("Attempt %d"%(attempts))
        accy = accuracy(dataset, final_weight, final_bias, target)
        #print("Accuracy %d"%(accy))

        if accy == 1 or attempts >= 10:
            break

        for element in range(len(dataset)):
            guess = prediction(dataset[element], final_weight, bias)

            # Actualize weights with error. Only actualize bias if flag is activated
            if learn_bias: final_bias += learning_factor * final_bias * (target[element] - guess)

            for entry in range(len(dataset[element])):
                delta_weight = learning_factor * dataset[element][entry] * (target[element] - guess)
                final_weight[entry] += delta_weight
        #print("Bias: %d" %(final_bias))
        #print("Weight:")
        #print(final_weight)

    return final_bias,final_weight

def separate_group(data, target_list, reference):
    '''
    This function is used to separate the "data" into smaller arrays according to the target group choosed in "reference".

    Example:
        data = [[1,2,3,4],[1,2,3,4]]
        target_list = [0,0,1,1]
        reference = 0
        group = separate_group(data, target_list, reference)
    '''
    data_group = np.empty((0,len(data[0])))
    for element in range(len(target_list)):
        if target_list[element] == reference:
            data_group = np.append(data_group, np.asmatrix(data[element]), axis=0)
    return data_group

def exercise_1():
    print("==========================================")
    print("Exercise 1: 1 neuron - Setosa x Versicolor")
    print("==========================================")
    print()

    iris = datasets.load_iris()

    setosa = separate_group(iris["data"], iris["target"], 0)
    versicolor = separate_group(iris["data"], iris["target"], 1)
    dataset = np.append(setosa, versicolor, axis=0)
    dataset = dataset.getA()

    targets = np.append([0 for x in range(len(setosa))], [1 for x in range(len(versicolor))])
    weight = np.array([1 for x in range(len(dataset[0]))])
    learning_factor = 2
    bias = 0.5

    accy_before = accuracy(dataset, weight, bias, targets)

    new_bias,new_weight = training(dataset, targets, bias, weight, learning_factor, learn_bias=True)

    accy_after = accuracy(dataset, new_weight, new_bias, targets)

    print()
    print("Weight:")
    print(new_weight)
    print("Bias: %f" %(new_bias))
    print("\nAccuracy before learning: %f\nAccuracy after learning: %f\n" %(accy_before,accy_after))

def exercise_2():
    print("====================================")
    print("Exercise 2: Single layer - 3 neurons")
    print("====================================")
    print()

    iris = datasets.load_iris()

    setosa = separate_group(iris["data"], iris["target"], 0)
    versicolor = separate_group(iris["data"], iris["target"], 1)
    virginica = separate_group(iris["data"], iris["target"], 2)

    dataset = np.append(setosa, versicolor, axis=0)
    dataset = np.append(dataset, virginica, axis=0)
    dataset = dataset.getA()
    
    #level 1 - setosa
    level1_targets = np.append([1 for x in range(len(setosa))], [0 for x in range(len(versicolor))])
    level1_targets = np.append(level1_targets, [0 for x in range(len(virginica))])
    weight = np.array([1 for x in range(len(dataset[0]))])
    learning_factor = 1
    level1_bias = 2

    level1_bias,level1_weight = training(dataset, level1_targets, level1_bias, weight, learning_factor, learn_bias=True)
    level1_accy_after = accuracy(dataset, level1_weight, level1_bias, level1_targets)

    #level 2 - virginica
    level2_targets = np.append([0 for x in range(len(setosa))], [0 for x in range(len(versicolor))])
    level2_targets = np.append(level2_targets, [1 for x in range(len(virginica))])
    weight = np.array([1 for x in range(len(dataset[0]))])
    learning_factor = 1
    level2_bias = 0

    level2_bias,level2_weight = training(dataset, level2_targets, level2_bias, weight, learning_factor, learn_bias=True)
    level2_accy_after = accuracy(dataset, level2_weight, level2_bias, level2_targets)

    #level 3 - versicolor
    level3_targets = np.append([0 for x in range(len(setosa))], [1 for x in range(len(versicolor))])
    level3_targets = np.append(level3_targets, [0 for x in range(len(virginica))])
    weight = np.array([1 for x in range(len(dataset[0]))])
    learning_factor = 2
    level3_bias = 1

    level3_bias,level3_weight = training(dataset, level3_targets, level3_bias, weight, learning_factor, learn_bias=True)
    level3_accy_after = accuracy(dataset, level3_weight, level3_bias, level3_targets)

    print()
    print("Weight 1:")
    print(level1_weight)
    print("Bias 1: %f" %(level1_bias))
    print("Accuracy: %f" %(level1_accy_after))
    print()
    print("Weight 2:")
    print(level2_weight)
    print("Bias 2: %f" %(level2_bias))
    print("Accuracy: %f" %(level2_accy_after))
    print()
    print("Weight 3:")
    print(level3_weight)
    print("Bias 3: %f" %(level3_bias))
    print("Accuracy: %f" %(level3_accy_after))
    print()

def main():
    exercise_1()
    exercise_2()


if __name__ == "__main__":
    main()