from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, average_precision_score
from pandas_confusion import ConfusionMatrix
import numpy as np
import pandas as pd
from tabulate import tabulate
import id3_tree

def findPath(graph ,start , end, pathSoFar):
    
    pathSoFar = pathSoFar + [start]
    if start == end :
        return pathSoFar
    if start not in graph:
        return None

    for node in graph[start]:
        if node not in pathSoFar:
            newpath = findPath(graph, node, end, pathSoFar)
            return newpath
    return None


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

def calc_precision_recall(y_actual,y_predict):
    
    TP, FP, TN, FN = perf_measure(y_actual,y_predict)

    average_precision = TP/float(TP+FP)

    average_recall = TP/float(TP+FN)

    return average_precision, average_recall



def main():
    
    filename = "mushroom.csv"
    
    idtree = id3_tree.id3_tree()
    data, classes, feature_names = idtree.read_data(filename)
    data, classes = shuffle(data, classes, random_state = 0)
   
 
    ''' 
    Split mushroom data into training and test data set in the ration of 
    3/4 training and 1/4 test dataset
    '''
    ntrain = int(0.85 * np.shape(data)[0])
   
    train_data = data[:ntrain]
    test_data = data[ntrain:]
    train_classes = classes[:ntrain]
    target_actual = classes[ntrain:]
    
    

    print "\n\nThe data split into train data: (%s, %s)" % np.shape(train_data)
    print "and Test dataset: : (%s, % s) \n\n" % np.shape(test_data)
    mushroom_feat_infogain = []
    feature = np.shape(feature_names)[0]
    feature_entropy = []  
    for i in range(feature):
        feature_entropy = idtree.calc_information_gain(train_data, train_classes, i)
    
    for i in range(feature):
    		mushroom_feat_infogain.append([feature_names[i],feature_entropy[i]])
    
    df = pd.DataFrame(mushroom_feat_infogain,columns=["Feature Name","Information Gain"])
    print  tabulate(mushroom_feat_infogain, headers=("Feature Name", "Information Gain"),
		tablefmt="orgtbl") + "\n \n \n "

    mtree =idtree.make_tree(train_data, train_classes, feature_names)
    idtree.printTree(mtree, ' ')

    
    target_predict= idtree.classifyAll(mtree, test_data)
           
  

    testClassifaiction = []
    for i in range(len(test_data)):
        testClassifaiction = np.append(testClassifaiction,idtree.classify(mtree, test_data[i]))

    confusion_matrix = ConfusionMatrix(target_actual, target_predict)
    print("Confusion matrix:\n\n%s" % confusion_matrix)

    
    accuracyScore = accuracy_score(target_actual, target_predict)
    print "\n Accurecy Score: %0.2f " % (accuracyScore * 100) + "%"

    '''
    calculating the precision and recall from the confusion matrix
    '''
    precision, recall = calc_precision_recall(target_actual,target_predict)

    print "precsion : %.2f and recall: %.2f \n\n" %(precision,recall)


if __name__ == "__main__":
      main()
