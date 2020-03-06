#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 01:19:27 2020

@author: apple
"""


from __future__ import print_function

import sys
from operator import add

from pyspark import SparkContext
import re
import numpy as np
from numpy import dot
from numpy.linalg import norm


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: task1 <file> <output> ", file=sys.stderr)
        exit(-1)
        
    sc = SparkContext()
    
    wikiCategoryLinks = sc.textFile(sys.argv[1],1)
    wikiCats=wikiCategoryLinks.map(lambda x: x.split(",")).map(lambda x: (x[0].replace('"', ''), x[1].replace('"', '') ))
    
    
    wikiPages = sc.textFile(sys.argv[2],1)
    
    # Assumption: Each document is stored in one line of the text file
    # We need this count later ... 
    numberOfDocs = wikiPages.count()
    
    # Each entry in validLines will be a line from the text file
    validLines = wikiPages.filter(lambda x : 'id' in x and 'url=' in x)
    
    # Now, we transform it into a set of (docID, text) pairs
    keyAndText = validLines.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6])) 
    
    def buildArray(listOfIndices):
        returnVal = np.zeros(20000)
        for index in listOfIndices: 
            returnVal[index] = returnVal[index] + 1

        mysum = np.sum(returnVal)
        returnVal = np.divide(returnVal, mysum) 
        return returnVal.tolist()

    # Cosine Similarity of two vectors
    def cousinSim (x, y):
        normA = np.linalg.norm(x)
        normB = np.linalg.norm(y) 
        return np.dot(x,y)/(normA*normB)
    
    regex = re.compile('[^a-zA-Z]')

    # remove all non letter characters
    keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))
    
    # Now get the top 20,000 words... first change (docID, ["word1", "word2", "word3", ...])to ("word1", 1) ("word2", 1)...
    allWords = keyAndListOfWords.flatMap(lambda x: x[1]).map(lambda x: (x,1))

    # Now, count all of the words, giving us ("word1", 1433), ("word2", 3423423), etc.
    allCounts = allWords.reduceByKey(add)
    
    # Get the top 20,000 words in a local array in a sorted format based on frequency
    topWords = allCounts.top(20000, key = lambda x: x[1])
    
    #allCounts_sort = allCounts.sortBy(lambda x: x[1],ascending = False)
    topTwenty = allCounts.top(20, key=lambda x: x[1])
    print("Top Words in Corpus:", topTwenty)
    
    # We'll create a RDD that has a set of (word, dictNum) pairs
    topWordsK = sc.parallelize(range(20000))

    # Now, we transform (0), (1), (2), ... to ("MostCommonWord", 1)

    dictionary = topWordsK.map (lambda x : (topWords[x][0], x))

    
    #dictionary_sort = dictionary.sortBy(lambda x: x[1],ascending = False)
    lastTwenty = dictionary.top(20, lambda x: x[1])
    print("Word Postions in our Feature Matrix. Last 20 words in 20k positions: ", lastTwenty)
    
    
    sc.parallelize(topTwenty,1).saveAsTextFile(sys.argv[3])
    sc.parallelize(lastTwenty,1).saveAsTextFile(sys.argv[4])


    sc.stop()