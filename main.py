import id3_tree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from collections import Counter, defaultdict

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

def make_tree(data, classes, feature_names):
    
    nData = len(data)
    num_features = len(data[0])
    new_classes = np.unique(classes)
    frequency = np.zeros(len(new_classes))
    

    totalEntropy = 0
    index=0
    for aclass in new_classes:
        frequency[index] = classes.count(aclass)
        totalEntropy += id3_tree.id3_tree().calc_entropy(float(frequency[index])/nData)

        index +=1
    default = classes[np.argmax(frequency)]
    if nData==0 or num_features==0:
        return default
    elif classes.count(classes[0])==nData:
        return classes[0]
    else:
        gain = np.zeros(num_features)
        featureSet = range(num_features)
        for feature in featureSet:
            g = id3_tree.id3_tree().calc_info_gain(data,classes,feature)
            gain[feature] = totalEntropy - g
        bestFeature = np.argmax(gain)
        tree = {feature_names[bestFeature]:{}}

        
        values = []
        feature=0
        for datapoint in data:
            if datapoint[feature] not in values:
                values.append(datapoint[bestFeature])
            feature += 1
        for value in values:
            newData = []
            new_classes = []
            index = 0
            for datapoint in data:
                if datapoint[bestFeature]==value:
                    if bestFeature==0:
                        datapoint = datapoint[1:]
                        newNames = feature_names[1:]
                    elif bestFeature==num_features:
                        datapoint = datapoint[:-1]
                        newNames = feature_names[:-1]
                    else:
                        datapoint = datapoint[:bestFeature]
                        datapoint.extend(datapoint[bestFeature+1:])
                        newNames = feature_names[:bestFeature]
                        newNames.extend(feature_names[bestFeature+1])
                    newData.append(datapoint)
                    new_classes.append(classes[index])
                index += 1
            

            subtree = make_tree(newData, new_classes, newNames)

            tree[feature_names[bestFeature]][value] = subtree
        
        return tree

            
        




def main():
    
    filename = "mushroomTest.csv"
    
    idtree = id3_tree.id3_tree()
    data, classes, feature_names = idtree.read_data(filename)
    
    mushroom_feat_infogain = []
    feature = np.shape(feature_names)[0]
    feature_entropy = []  
    for i in range(feature):
        feature_entropy= idtree.calc_information_gain(data, classes, i)
    
    for i in range(feature):
    		mushroom_feat_infogain.append([feature_names[i],feature_entropy[i]])
    
    df = pd.DataFrame(mushroom_feat_infogain,columns=["Feature Name","Information Gain"])
    print  tabulate(mushroom_feat_infogain, headers=("Feature Name", "Information Gain"),
		tablefmt="orgtbl")

    mtree =idtree.make_tree(data, classes, feature_names)
    idtree.printTree(mtree, ' ')
    print idtree.classifyAll(mtree, data)

    for i in range(len(data)):
       classified = idtree.classify(mtree, data[i])
    
    print classified


          





if __name__ == "__main__":
      main()
