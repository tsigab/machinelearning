from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.model_selection import KFold
from pandas_confusion import ConfusionMatrix
from collections import Counter
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from tabulate import tabulate
import id3_tree


def perf_measure(y_actual, y_predict):
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for i in range(len(y_predict)):
        if y_actual[i] == y_predict[i] == 'e':
               true_positive += 1
        elif y_predict[i] == 'e' and y_actual[i] != y_predict[i]:
           false_positive += 1
        elif y_actual[i] == y_predict[i] == 'p':
           true_negative += 1
        else:
           false_negative += 1
    return(true_positive, false_positive, true_negative, false_negative)

def tree_depth(tree, level=1):
    if not isinstance(tree, dict) or not tree:
        depth = lambda level : level if (level % 2==0) else (level/2)+1
        return depth(level)

    return max(tree_depth(tree[i], level + 1) for i in tree)

def calc_precision_recall(y_actual,y_predict):
    
    TP, FP, TN, FN = perf_measure(y_actual,y_predict)

    average_precision = TP/float(TP+FP)

    average_recall = TP/float(TP+FN)

    return average_precision, average_recall




def early_stopping(train_data, train_classes, test_data, test_target, feature_names):
    
    depth = []
    accuracyscore = []
    prediction_test = []
    performace_depth = []
    tree_depth = 0
    idtree = id3_tree.id3_tree()
    for i in range(len(feature_names)):
        mtree = idtree.make_tree(train_data, train_classes, feature_names, maxlevel=tree_depth)
        predicted_output = idtree.classifyAll(mtree, test_data)
        accuracyscore = np.append(accuracy_score, accuracy_score(test_target, predicted_output))
        
        TP, FP, TN, FN = perf_measure(test_target, predicted_output)

        performace_depth.append([i,TP, FP, TN, FN])
        depth.append(tree_depth)
        tree_depth += 1
        
    return accuracyscore, performace_depth
    



def main():
    
    filename = "mushroom.csv"
    
    idtree = id3_tree.id3_tree()

    '''
    split the data into data features, output classes and label names.
    '''
    data, classes, feature_names = idtree.read_data(filename)
    data, classes = shuffle(data, classes, random_state = 0)
   
    target_label = dict(Counter(classes))

    print "target datase contains Poisonous: %d and Edible: %d" % (target_label.values()[0], target_label.values()[1])
    ''' 
    Split mushroom data into training and test data set  
    3/4 training and 1/4 test dataset
    '''
    ntrain = int(0.75 * np.shape(data)[0])

       
    train_data = data[:ntrain]
    test_data = data[ntrain:]
    train_target = classes[:ntrain]
    test_target = classes[ntrain:]
    
    

    print "\n\nThe data split into train data: (%s, %s)" % np.shape(train_data)
    print "and Test dataset: : (%s, % s) \n\n" % np.shape(test_data)
    
    
    feature_info_gain = []
    feature = np.shape(feature_names)[0]
    feature_entropy = []  
    for i in range(feature):
        feature_entropy = idtree.calc_information_gain(data, classes, i)
    
    for i in range(feature):
    		feature_info_gain.append([feature_names[i],feature_entropy[i]])
    print "Information gain for the whole features in mushroom dataset\n"
    df = pd.DataFrame(feature_info_gain)   
    print  tabulate(feature_info_gain, headers=("Feature Name", "Information Gain"),
		tablefmt="orgtbl") + "\n \n \n "

    '''
    train the dataset and creating tree from the training data
    '''
    mtree = idtree.make_tree(train_data, train_target, feature_names,maxlevel=4)

    idtree.printTree(mtree,'')
    
    predicted_output = idtree.classifyAll(mtree, test_data)
    confusion_matrix = ConfusionMatrix(test_target, predicted_output)
    print("\nConfusion matrix:\n\n%s" % confusion_matrix)
    
    accuracyScore = accuracy_score(test_target, predicted_output)
    print "\nAccurecy Score: %0.2f " % (accuracyScore * 100) + "%"

    '''
    calculating the precision and recall from the confusion matrix
    '''
    precision, recall = calc_precision_recall(test_target, predicted_output)

    print "\nprecision and recall of the classifier is: \nPrecision %0.2f \nRecall %0.4f " % (precision, recall)

    print 
    accuracyscores, perf_measure = early_stopping(train_data, train_target, test_data,test_target, feature_names)

    print "Performance of the classifier with early stopping"
    print tabulate(perf_measure[:6], headers=( "TP", "FP", "TN", "FN"),
                   tablefmt="orgtbl") + "\n \n \n "

if __name__ == "__main__":
      main()
