#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 01:57:46 2020

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
        print("Usage: task2 <file> <output> ", file=sys.stderr)
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
        return returnVal

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
    
    # We'll create a RDD that has a set of (word, dictNum) pairs start by creating an RDD that has the number 0 through 20000
    # 20000 is the number of words that will be in our dictionary
    topWordsK = sc.parallelize(range(20000))

    # Now, we transform (0), (1), (2), ... to ("MostCommonWord", 1)
  
    dictionary = topWordsK.map (lambda x : (topWords[x][0], x))

    
    # Next, we get a RDD that has, for each (docID, ["word1", "word2", "word3", ...]),("word1", docID), ("word2", docId), ...
    
    allWordsWithDocID = keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))
    
    # Now join and link them, to get a set of ("word1", (dictionaryPos, docID)) pairs
    allDictionaryWords = allWordsWithDocID.join(dictionary)

    # Now, we drop the actual word itself to get a set of (docID, dictionaryPos) pairs
    justDocAndPos = allDictionaryWords.map(lambda x : x[1])
    
    # Now get a set of (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
    allDictionaryWordsInEachDoc = justDocAndPos.groupByKey()
    
    # The following line this gets us a set of (docID,  [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs

    allDocsAsNumpyArrays = allDictionaryWordsInEachDoc.map(lambda x: (x[0], buildArray(x[1])))
    
    result1 = allDocsAsNumpyArrays.take(3)
    
    sc.parallelize(result1,1).saveAsTextFile(sys.argv[3])
    
    # Now, create a version of allDocsAsNumpyArrays 
    def zeroOne(arr):
      arr[arr!=0] = 1
      return arr
      
    zeroOrOne = allDocsAsNumpyArrays.map(lambda x : (x[0],zeroOne(x[1])))

    
    # Now, add up all of those arrays into a single array
    dfArray = zeroOrOne.reduce(lambda x1, x2: ("", np.add(x1[1], x2[1])))[1]
    
    # Create an array of 20,000 entries, each entry with the value numberOfDocs (number of docs)
    multiplier = np.full(20000, numberOfDocs) #1000
    
    # Get the version of dfArray where the i^th entry is the inverse-document frequency for the
    # i^th word in the corpus
    idfArray = np.log(np.divide(np.full(20000, numberOfDocs), dfArray)) #idf
    
    
    # Finally, convert all of the tf vectors in allDocsAsNumpyArrays to tf * idf vectors
    allDocsAsNumpyArraysTFidf = allDocsAsNumpyArrays.map(lambda x: (x[0], np.multiply(x[1], idfArray)))
    
    result2 = allDocsAsNumpyArraysTFidf.take(2)

    
    sc.parallelize(result2,1).saveAsTextFile(sys.argv[4])
    
    
    sc.stop()