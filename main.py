from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from pandas_confusion import ConfusionMatrix

from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def main():
    
    filename = "mushroom.csv"
    
    idtree = id3_tree.id3_tree()
    data, classes, feature_names = idtree.read_data(filename)
    
    
    np.random.seed(750)
    data, classes = shuffle(data, classes, random_state = 0)
   
 

    ntrain = int(0.75 * np.shape(data)[0])
   
    train_data = data[:ntrain]
    test_data = data[ntrain:]
    train_classes = classes[:ntrain]
    test_classes = classes[ntrain:]
    
    

    print "The data split into train data: %s and test data %s:" % np.shape(train_data)
    print np.shape(test_data) ,np.shape(test_classes)
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

    
    predict= idtree.classifyAll(mtree, test_data)
           
  

    testClassifaictin = []
    for i in range(len(test_data)):
        testClassifaictin = np.append(testClassifaictin,idtree.classify(mtree, test_data[i]))

    
    

    
    confusion_matrix = ConfusionMatrix(test_classes, predict)
    print("Confusion matrix:\n\n%s" % confusion_matrix)
    

    accuracyScore = accuracy_score(test_classes,predict)
    print "Accurecy Score: %0.2f " % (accuracyScore*100) + "%"





if __name__ == "__main__":
      main()
