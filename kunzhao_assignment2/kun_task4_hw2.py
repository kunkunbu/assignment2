#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 03:56:57 2020

@author: apple
"""


from __future__ import print_function

from pyspark.sql import SparkSession
from pyspark import SparkContext


import sys

from pyspark import SparkConf,SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql.types import ArrayType, FloatType, StringType, IntegerType
import requests

from pyspark.sql.types import *
from pyspark.sql import functions as func

from pyspark.sql.functions import lit
from pyspark.sql.functions import udf
from pyspark.sql.functions import *
from pyspark.sql.functions import array
from pyspark.sql.functions import arrays_zip
from pyspark.sql import SQLContext

#spark = SparkSession.builder.master("local[*]").getOrCreate()

from pyspark.sql import Row

import re
import numpy as np
from numpy import dot
from numpy.linalg import norm
from pyspark.sql.functions import split, explode
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml.linalg import Vectors, VectorUDT
#spark = SparkSession.builder.master("local[*]").getOrCreate()


if __name__ == "__main__":

        
    sc = SparkContext()
    sqlContext = SQLContext(sc)
    
    wikiCategoryLinks = sqlContext.read.format('csv').options(sep = ",").load(sys.argv[1])
    wikiCategoryLinks.cache()

    wikiCats = wikiCategoryLinks.selectExpr("_c0 as docID", "_c1 as category")
    wikiCats.cache()

    wikiPages = sqlContext.read.format('csv').options(sep="|").load(sys.argv[2])
    wikiPages.cache()

    numberOfDocs = wikiPages.count()
    
    getID_udf = udf(lambda x: x[x.index('id="') + 4 : x.index('" url=')],StringType())
    getText_udf = udf(lambda x: x[x.index('">') + 2:][:-6],StringType())
    
    docIDAndText = wikiPages.withColumn("docID", getID_udf("_c0")).withColumn("text", getText_udf("_c0")).drop("_c0")
    docIDAndText.cache()

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
    textToWord_udf = udf(lambda x : str(regex.sub(' ', x).lower().split())[1:-1],StringType())
    
    docIDAndListOfWords = docIDAndText.withColumn("ListofWords", textToWord_udf("text")).drop("text")
    docIDAndListOfWords.cache()
    
    allWords = docIDAndListOfWords.withColumn('word',explode(split('ListofWords',','))).withColumn('count',lit(1)).drop("ListofWords")
    allWords.cache()


    allCounts = allWords.groupBy("word").agg(func.sum("count"))
    allCounts.cache()


    # Get the top 20,000 words in a local array in a sorted format based on frequency
    topWords = allCounts.orderBy("sum(COUNT)", ascending=False).limit(20000)
    topWords.cache()

    
    #topTwenty = allCounts.orderBy("sum(COUNT)", ascending=False).limit(20)

    
    
    #two columns: word and position
    
    dictionary =topWords.withColumn("position",monotonically_increasing_id()).drop("sum(count)")
    dictionary.cache()

    #lastTwenty = dictionary.orderBy("position",ascending=False).limit(20)
    
   
    
    #print("Word Postions in our Feature Matrix. Last 20 words in 20k positions: ",lastTwenty.show())
    #sc.parallelize(topTwenty,1).saveAsTextFile(sys.argv[4])
    
    
    #----------------*************task 2***********---------------
    
    #columns: docID, word
    allWordsWithDocID = allWords.drop("count")
    allWordsWithDocID.cache()
    
    #columns:docID, word, position
    allDictionaryWords = allWordsWithDocID.join(dictionary, allWordsWithDocID.word == dictionary.word,"inner") 
    allDictionaryWords.cache()
    
    justDocAndPos = allDictionaryWords.drop("word") #columns: docID, position
    justDocAndPos.cache()
    
    #columns: docID and a list of positions
    allDictionaryWordsInEachDoc = justDocAndPos.groupBy("docID").agg(func.collect_list(func.col("position")).alias("position"))
    allDictionaryWordsInEachDoc.cache()

    #get tf array, columns: docID and tf
    buildArray_udf = udf(lambda x: buildArray(x), ArrayType(FloatType()))
    
    
    allDocsAsNumpyArrays = allDictionaryWordsInEachDoc.withColumn("tf",buildArray_udf("position")).drop("position")
    allDocsAsNumpyArrays.cache()
    

    # Now, create a version of allDocsAsNumpyArrays where, in the array,
    # every entry is either zero or one.

    
    #two columns:docID and zeroOrOne

    zeroOne_udf = udf(lambda x: np.clip(np.multiply(np.array(x),30000),0,1).tolist(),ArrayType(FloatType()))
    
    zeroOrOne = allDocsAsNumpyArrays.withColumn("tf_zeroOne", zeroOne_udf("tf")).drop("tf")

    zeroOrOne.cache()
    
    zeroOrOne_new = zeroOrOne.withColumn("new_key",lit(1))

    zeroOrOne_new.cache()



    def add_list(x):
      all = np.sum(Vectors.dense(x))
      return all

    add_udf = udf(add_list,ArrayType(FloatType()))

    zeroOrOne_final = zeroOrOne_new.groupBy("new_key").agg(add_udf(func.collect_list("tf_zeroOne")).alias("sum")).drop("new_key")
    #zeroOrOne_final = zeroOrOne_new.groupBy("new_key").agg(func.sum(func.collect_list("tf_zeroOne")).alias("sum")).drop("new_key")


    dfArray = zeroOrOne_final.select("sum").first()[0]
 
    
    
    print(dfArray)

    # Now, add up all of those arrays into a single array, where the
    # i^th entry tells us how many
    # individual documents the i^th word in the dictionary appeared in
    


    #n = len(zeroOrOne_new.select('tf_zeroOne').first()[0])

    #zeroOrOne_rdd = zeroOrOne.rdd
    #zeroOrOne_rdd.cache()

    #dfArray = zeroOrOne_final.collect()
    #dfArray = zeroOrOne_rdd.reduce(lambda x1, x2: ("", np.add(x1[1], x2[1])))[1]
    #zeroOrOne_array = np.array(zeroOrOne.select("tf_zeroOne").collect())
    #zeroOrOne_array.cache()
    #dfArray = np.sum(zeroOrOne_array,0).tolist()


    # Create an array of 20,000 entries, each entry with the value numberOfDocs (number of docs)
    multiplier = np.full(20000, numberOfDocs) #1000

    # Get the version of dfArray where the i^th entry is the inverse-document frequency for the
    # i^th word in the corpus
    idfArray = np.log(np.divide(np.full(20000, numberOfDocs), dfArray)) #idf
    
    # Finally, convert all of the tf vectors in allDocsAsNumpyArrays to tf * idf vectors
    tfidf_udf = udf(lambda x: np.multiply(np.array(x), idfArray).tolist(), ArrayType(FloatType()))
    allDocsAsNumpyArraysTFidf = allDocsAsNumpyArrays.withColumn("Tf-Idf", tfidf_udf("tf") ).drop("tf")
    allDocsAsNumpyArraysTFidf.cache()
    
    #print(allDocsAsNumpyArraysTFidf.show(2))
    
    #----------------*************task 3***********---------------

    # now we join the wiki categories with the wiki page numpy array tf-idf and generate many rows from one wiki page 
    # based on number of category that a wiki page is in it. 
    
    wikiAndCatsJoind = wikiCats.join(allDocsAsNumpyArraysTFidf, wikiCats.docID == allDocsAsNumpyArraysTFidf.docID, "inner")

    wikiAndCatsJoind.cache()

    features =  wikiAndCatsJoind.drop("docID")

    features.cache()
    # Finally, we have a function that returns the prediction for the label of a string, using a kNN algorithm
    
    #rdd = sc.parallelize([Row(a=1,b=2,c=3),Row(a=4,b=5,c=6),Row(a=7,b=8,c=9)])
    
    def getPrediction (textInput, k):
        # Create an RDD out of the textIput
        myDocRdd = sc.parallelize ([Row(text = textInput)])
    
        
        myDoc = myDocRdd.toDF()
    
        # Flat map the text to (word, 1) pair for each word in the doc
    
        #textToWord_udf = udf(lambda x : str(regex.sub(' ', x).lower().split())[1:-1],StringType())
    
        listofWords = myDoc.withColumn("ListofWords", textToWord_udf("text")).drop("text")
    
    
        wordsInThatDoc = listofWords.withColumn('word',explode(split('ListofWords',','))).withColumn('count',lit(1)).drop("ListofWords")
       
     
        # This will give us a set of (word, (dictionaryPos, 1)) pairs
        allDictionaryWordsInThatDoc = dictionary.join (wordsInThatDoc, dictionary.word == wordsInThatDoc.word,"inner")
        
        #print(allDictionaryWordsInThatDoc.show())
    
        # Get tf array for the input string
        #print(allDictionaryWordsInThatDoc.top (1)[0][1]) #[pos1, pos2, pos1, pos4, pos5,.....]
        
    
        buildArray_udf = udf(lambda x: buildArray(x), ArrayType(FloatType()))
        
    
        myArray_df = allDictionaryWordsInThatDoc.groupBy(wordsInThatDoc.word)\
                                                          .agg(func.sum("count").alias("count"),buildArray_udf(func.collect_list("position")).alias("idf"))\
                                                          .orderBy("count",ascending=False).limit(1)
        
    
        myArray = myArray_df.select("idf").collect()
    
    
        # Get the tf * idf array for the input string
        myTfidf = np.multiply (np.array(myArray)[0][0],idfArray)
    
        
        # Get the distance from the input text string to all database documents, using cosine similarity (np.dot() )
        distance_udf = udf(lambda x: float(np.dot (np.array(x), myTfidf)),FloatType())
        
        distances = features.withColumn("distance", distance_udf("Tf-Idf"))
        distances.cache()
 
    
        # get the top k distances
    
        topK = distances.orderBy("distance", ascending=False).limit(k) #category, Tf-Idf, distance
 
    
        # and transform the top k distances into a set of (catogory, 1) pairs
        
        catRepresented = topK.withColumn("rank",lit(1)).drop("Tf-Idf") #(category,1)
 
    
        # now, for each category, get the count of the number of times this category appeared in the top k
     
        numTimes = catRepresented.groupBy("category").agg(func.sum("rank").alias("num")).drop("rank")
        
        numTimes_ordered = numTimes.orderBy("num", ascending=False).limit(k)
    
        # Return the top k of them.
        return numTimes_ordered.collect()

    
    text1_result = getPrediction('Sport Basketball Volleyball Soccer', 10)
    text2_result = getPrediction('What is the capital city of Australia?', 10)
    text3_result = getPrediction('How many goals Vancouver score last year?', 10)
    
    print(text1_result)
    print(text2_result)
    print(text3_result)    
    #text1_result.format("text").option("header", "false").mode("append").save(sys.argv[3])  
    #text2_result.format("text").option("header", "false").mode("append").save(sys.argv[4]) 
    #text3_result.format("text").option("header", "false").mode("append").save(sys.argv[5]) 
    
    sc.parallelize(text1_result,1).saveAsTextFile(sys.argv[3])
    sc.parallelize(text2_result,1).saveAsTextFile(sys.argv[4])
    sc.parallelize(text3_result,1).saveAsTextFile(sys.argv[5])
    
    sc.stop()
