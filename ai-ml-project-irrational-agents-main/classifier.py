# Filename: classifier.py
# Author: Tom Sullivan and Jack Hoy
# Dates: 12/02/2022, 12/07/2022, 12/08/2022/, 12/09/2022

import sys
import csv
import itertools
import math
import scipy.stats

#begin Node class
#_____________________________________________________________________________________________________________________________________
class Node:
    def __init__(self, attribute, children = {}, parent = None):
        self.parent = parent
        self.children = children
        self.attribute = attribute #Used for root test  
#_____________________________________________________________________________________________________________________________________
#end Node class

'''#begin printTree
#_____________________________________________________________________________________________________________________________________'''
def printTree(rootNode, test_examples, p_type, n_type, attributes):#print tree simply prints a lot of stats about the decision tree, print tree 2 recursively prints out the tree structure
    totalNodes, decisionNodes, maxD, minD, dS, TP, TN, FP, FN = printTree2(rootNode, 0, 1, 0, 0, 0, 0, test_examples, p_type, n_type, attributes,0,0,0,0)
    print("Total Nodes: " + str(totalNodes))
    print("Decision Nodes: " + str(decisionNodes))
    print("Maximum Depth: " + str(maxD))
    print("Minimum Depth: " + str(minD))
    print("Average Depth of Root-to-Leaf: " + str(dS/decisionNodes))
    print('\n')
    print("Testing decision tree with test set:")
    print("True " + str(p_type) + ": " + str(TP))
    print("True " + str(n_type) + ": " + str(TN))
    print("False " + str(p_type) + ": " + str(FP))
    print("False " + str(n_type) + ": " + str(FN))
    print("Recognition Rate: " + str(recognitionRate(TP, TN, FP, FN)))

'''#_____________________________________________________________________________________________________________________________________
#end printTree'''




'''#begin printTree2
#_____________________________________________________________________________________________________________________________________'''
def printTree2(rootNode, depth, totalNodes, decisionNodes, maxD, minD, depthSum, test_examples, p_type, n_type, attributes,TP, TN, FP, FN):
    minD = 1
    if rootNode.children == {}:#base case for when tree is just a single leaf node
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        if rootNode == p_type:
            for tex in test_examples:
                if tex[attributes[0][0]] == p_type:
                    TP +=1
                else:
                    FP += 1
        return 1, 1, 0, 0, 0, TP, 0, FP, 0

    for i in range(depth):
        print('\t', end = " ")
    print("Testing " + rootNode.attribute[0]) #print all non-leaf nodes
    attributeVals = rootNode.attribute.copy()
    attributeVals.pop(0)#create a list of attribute values, without the attribute name

    for branch in attributeVals:#loop through all attribute values. e.g. y, n, ?
        branch_examples = []
        for tex in test_examples:
            if tex[rootNode.attribute[0]] == branch:
                branch_examples.append(tex)#branch examples is the subset of examples with the given attribute.
        for i in range(depth+1):
            print('\t', end = " ") #print spaces for how deep in the tree we are.
        print("Branch " + branch)
        
        if not isinstance(rootNode.children[branch],str): #if not a leaf node
            tN, dN, mD, miD, dS, tp, tn, fp, fn = printTree2(rootNode.children[branch], depth + 1, totalNodes + 1, decisionNodes, maxD, 1, depthSum, branch_examples, p_type, n_type, attributes, TP, TN, FP, FN) #read into temp variables to pass up the tree
            maxD = mD
            totalNodes = tN
            depthSum = dS
            decisionNodes = dN
            TN = tn
            TP = tp
            FP = fp
            FN = fn
        else: #If you're at a leaf node
            TPk = 0
            FPk = 0
            TNk = 0
            FNk = 0
            for i in range(depth+1):
                print('\t', end = " ")
            print("Node with value " + str(rootNode.children[branch]))#print out value of the leaf node
            if rootNode.children[branch] == p_type:#If the node is of type p, compare to the test set data.
                for b in branch_examples:
                    if b[attributes[0][0]] == p_type:
                        TPk += 1#true positive
                    else:
                        FPk += 1#false positive
            else:
                for b in branch_examples:
                    if b[attributes[0][0]] == n_type:
                        TNk += 1#true negative
                    else:
                        FNk += 1#false negative

            #Sum up all true positives etc for confusion matrix
            TP += TPk
            FP += FPk
            TN += TNk
            FN += FNk
            totalNodes += 1
            decisionNodes += 1
            maxD = max(maxD, depth + 1)
            depthSum += depth + 1    
    return totalNodes, decisionNodes, maxD, minD, depthSum, TP, TN, FP, FN
'''#_____________________________________________________________________________________________________________________________________
#end printTree2'''


'''begin recognition rate
#_____________________________________________________________________________________________________________________________________'''
def recognitionRate(TP, TN, FP, FN):
    return float((TP + TN)/(TP + TN + FP + FN)) #This is recognition rate formula
'''_____________________________________________________________________________________________________________________________________
end recognition rate'''


#begin all equal
#_____________________________________________________________________________________________________________________________________
def all_equal(iterable):#From https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical, User: mgilson
    #Returns True if all the elements are equal to each other
    g = itertools.groupby(iterable)
    return next(g, True) and not next(g, False)
#_____________________________________________________________________________________________________________________________________
#end all equal


'''#begin plurality value
#_____________________________________________________________________________________________________________________________________'''
def PluralityVal(ex_classifications): #From https://stackoverflow.com/questions/20038011/trying-to-find-majority-element-in-a-list, user: FogleBird
    return max(set(ex_classifications),key = ex_classifications.count)#returns element of list that occurs the most times
'''#_____________________________________________________________________________________________________________________________________
#end plurality value'''


#begin entropy, known as 'B' in the notes. *Make sure to pass it a float instead of an int.*
#_____________________________________________________________________________________________________________________________________
def Entropy(input):
    q = float(input)
    if(input == 0 or input == 1):
        return 0
    return -1*(q*math.log2(q) + (1-q)*math.log2(1-q))#Entropy is simply this mathematical function
#_____________________________________________________________________________________________________________________________________
#end entropy


'''#begin remainder
#_____________________________________________________________________________________________________________________________________'''
def Remainder(examples, attribute, attributes, p_type, p, n):
    remainderVal = 0
    attributeVals = attribute.copy()
    attributeVals.pop(0)#remove name of attribute, resulting list only has the values that the attribute can take on. (branches in tree)
    
    for k in attributeVals:#find each type of value for a given attribute-- or the different branches.
        pk = 0
        nk = 0
        k_examples = []
        for ex in examples:
            if ex[attribute[0]] == k:
                k_examples.append(ex)#select examples that have that attribute value.
        for k_ex in k_examples:
            if k_ex[attributes[0][0]] == p_type:#out of our examples of a specific type, if the classification of the example matches p_type, then increment pk
                pk += 1
            else:
                nk += 1#same as pk, but for the n_type
        if(pk+nk == 0):#this should not happen, but if somehow it does this will prevent a crash
            remainderVal += 0
        else:
            remainderVal += (((pk + nk)/(p + n))*Entropy(pk/(pk + nk)))#remainder is this summation
    return remainderVal
'''#_____________________________________________________________________________________________________________________________________
#end remainder'''


#begin importance
#_____________________________________________________________________________________________________________________________________
def Importance(examples, attribute, attributes, p_type):
    p = 0
    n = 0
    for ex in examples:
        if ex[attributes[0][0]] == p_type:#find all p or n classifications in the whole example set
            p += 1
        else:
            n += 1
    return Entropy(p/(p + n)) - Remainder(examples, attribute, attributes, p_type, p, n)#Importance is just this mathematical formula

#_____________________________________________________________________________________________________________________________________
#end importance


'''#begin decision tree learning
#_____________________________________________________________________________________________________________________________________'''
def DecisionTreeLearning(examples, attributes, parent_examples, p_type):
    if parent_examples != []:
        p_classifications = []#p_classifications is a list of all of the final results that the parent node had. e.g. [republican, republican, democrat, democrat]
        for entry in parent_examples:
            p_classifications.append(entry[attributes[0][0]]) #This assumes that the attributes file lists the classification as the first attribute.

    classifications = []
    for entry in examples:
        classifications.append(entry[attributes[0][0]]) #classifications is a list of all the final classifications that the current examples set has. e.g. [republican, democrat]
    #Base cases
    if (examples == []):  #If there are no examples left, return what the parent node had the most classifications of.
        return PluralityVal(p_classifications)
    elif all_equal(classifications):#if there's only one type of final classification left, return that classification
        return classifications[0]
    elif attributes == []:#if there are no attributes left to test, return the most common classification remaining
        return PluralityVal(classifications)
    else:
        attributesNoClass = attributes.copy()
        attributesNoClass.pop(0) #exclude the classification attribute
        attrMax = attributesNoClass[0]#pick any attribute, this just happens to be the first one
        for attr in attributesNoClass:
            if Importance(examples,attr,attributes,p_type) > Importance(examples,attrMax,attributes,p_type):#if you find a more important attribute than the current most important, use that one as the new most important
                attrMax = attr #choose attribute with highest importance value.
        childrenDict = {}#prepare empty dictionary for child nodes
        attrMaxCopy= attrMax.copy()
        attrMaxCopy.pop(0)

        for k in attrMaxCopy:#iterate through the branches on the chosen attribute
            k_examples = []
            for ex in examples:
                if ex[attrMax[0]] == k:
                    k_examples.append(ex)#choose the subset of examples that has that value for the given attribute
            new_attributes = attributes.copy()
            new_attributes.remove(attrMax)#remove the current attribute from the available list of future attributes
            subtree = DecisionTreeLearning(k_examples, new_attributes, examples, p_type)#recursively create subtrees until the base cases.
            childrenDict.update({k:subtree})#add the head of the subtree to the list of child nodes on the current node.
        treeHead = Node(attrMax, childrenDict)#the current node, with all of the child nodes filled out.
        return treeHead

'''#_____________________________________________________________________________________________________________________________________
#end decision tree learning'''

#begin chi squared pruning
#_____________________________________________________________________________________________________________________________________
def chisquaredPrune(node, examples, attributes, p_type, significance):
    attributeVals = node.attribute.copy()
    attributeVals.pop(0)
    leafNodeCount = 0
    for k in attributeVals:#first loop, prune every node except top node. k is a branch off of the current node.
        k_examples = []
        for ex in examples:
            if ex[node.attribute[0]] == k:
                k_examples.append(ex)#find all examples with the value k for the current attribute
        if not isinstance(node.children[k],str): #if branch leads to another node
            node.children[k] = chisquaredPrune(node.children[k], k_examples, attributes, p_type, significance) #test if that node needs to be pruned
            if not isinstance(node.children[k],str):#if the node was not pruned, move on to other branches off of the current node.
                secondVals = attributeVals.copy()
                secondVals.remove(k)#the remaining branches, just without the branch k
                for j in secondVals:
                    j_examples = []
                    for ex2 in examples:
                        if ex2[node.attribute[0]] == k:
                            j_examples.append(ex2)#create a subset of examples that have the value j for the current attribute
                    if isinstance(node.children[j],Node):
                        node.children[j] = chisquaredPrune(node.children[j], j_examples, attributes, p_type, significance)#try to prune the node
                return node
            else:
                return chisquaredPrune(node, examples, attributes, p_type, significance)#If you pruned a node, try to prune yourself again.
        else: # if branch leads to leaf node
            leafNodeCount += 1
    if leafNodeCount == len(attributeVals):#If all branches lead to leaf nodes
        p = 0
        n = 0
        for ex in examples:
            if ex[attributes[0][0]] == p_type:#count number of p and n values in the current examples
                p += 1
            else:
                n += 1
        deviance = 0
        numBranches = 0 #actual branches we have
        for k in attributeVals:
            pk = 0
            nk = 0
            k_examples = []
            for ex in examples:
                if ex[node.attribute[0]] == k:
                    k_examples.append(ex)#create the list of examples that have the value k for a given attribute
            for k_ex in k_examples:
                if k_ex[attributes[0][0]] == p_type:#if the k_example has the classifier of type p, add to pk.
                    pk += 1
                else:
                    nk += 1#same as pk, but nk
            phatk = float(p*((pk + nk)/(p + n)))#p hat formula
            nhatk = float(n*((pk + nk)/(p + n)))#n hat formula
            if(pk != 0 or nk != 0):#if there are values for that attribute value, then there must be a branch
                numBranches += 1
            if(pk != 0):#one half of the deviance formula
                deviance += (((pk - phatk)*(pk - phatk))/phatk)
            if(nk != 0):#other half of the deviance formula
                deviance += (((nk - nhatk)*(nk - nhatk))/nhatk)
        degrees = numBranches - 1#degrees of freedom formula
        val = scipy.stats.chi2.ppf(1-significance, df = degrees)#chi squared value
        if val > deviance: #This is prune condition
            classifications = []
            for entry in examples:
                classifications.append(entry[attributes[0][0]])#If you prune the node, take the plurality of all the classifications from the node. In our function, this is a string.
            return PluralityVal(classifications)
        else:
            return node
#_____________________________________________________________________________________________________________________________________
#end chi squared pruning


#begin main
#_____________________________________________________________________________________________________________________________________
def main():
    if len(sys.argv) != 5 and len(sys.argv) != 4:
        print('Incorrect program arguments. Correct usage: python3 classifier.py <attributes> <training-set> <testing-set> <significance>')
        print('Inclusion of <significance> is optional.')
        return
    
    attributes_file = open(sys.argv[1]) 
    attributesStr = attributes_file.readlines() #read attributes file line by line
    attributes = []
    for i in range(0, len(attributesStr)):
        attributesStr[i] = attributesStr[i].strip()
        attributes.append(attributesStr[i].split(',')) #attributes is a 2D array, with the attribute followed by its values for each attribute

    training_set_file = open(sys.argv[2], newline='')#read in the training set from a csv file
    training_set = csv.DictReader(training_set_file)
    examples = []
    for row in training_set:
        examples.append(row) #examples is list of dictionaries

    test_set_file = open(sys.argv[3], newline='')
    test_set = csv.DictReader(test_set_file)#read in test set from a csv
    test_examples = []
    for row in test_set:
        test_examples.append(row) #test_examples is list of dictionaries

    p_type = attributes[0][1]#choose a value to be 'p' e.q. republican or yes
    n_type = attributes[0][2]#choose other value to be n
    decisionTreeHead = DecisionTreeLearning(examples, attributes, [], p_type)#create our decision tree

    if (len(sys.argv) == 5):#If we want to prune our tree
        significance = float(sys.argv[4])
        chisquaredPrune(decisionTreeHead, examples, attributes, p_type, significance)
    printTree(decisionTreeHead, test_examples, p_type, n_type, attributes)#prints the tree and relevant statistics
#_____________________________________________________________________________________________________________________________________
#end main

#For running in interpreter, for testing.
if __name__ == "__main__":
    main()