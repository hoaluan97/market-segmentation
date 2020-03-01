# import libs
import pyspark.sql
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoderEstimator
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
import datetime
from pyspark.ml.feature import PCA
from pyspark.sql import Row
import csv
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import entropy
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from math import sqrt
from math import log
from pyspark.sql import SparkSession

spark = SparkSession\
        .builder\
        .appName("PythonKMeans")\
        .getOrCreate()

def cosine_similarity(vector1, vector2):
        norm_v1 = np.linalg.norm(vector1)
        norm_v2 = np.linalg.norm(vector2)
        return ((np.dot(vector1, vector2))/(norm_v1 * norm_v2))
    
def pearson_similarity(vector1, vector2):
    v1 = vector1 - np.mean(vector1)
    v2 = vector2 - np.mean(vector2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return ((np.dot(v1, v2)) / (norm_v1 * norm_v2))

def closestPoint(p, centers, distanceMeasure):
    bestIndex = 0
    closest = float("+inf")
    for i in range(len(centers)):
        if (distanceMeasure == 'cosine'):
#         tempDist = np.sum((p[1] - centers[i]) ** 2)
            tempDist = 1 - cosine_similarity (p[1], centers[i])
#         tempDist = distance.cosine(p[1], centers[i])
        elif (distanceMeasure == 'pearson'):
            tempDist = 1 - pearson_similarity (p[1], centers[i])
        else:
            tempDist = np.sum((p[1] - centers[i]) ** 2)


        if tempDist < closest:
            closest = tempDist
            bestIndex = i
    return bestIndex

def error(p, kPoints, distanceMeasure):
    center = kPoints[closestPoint(p, kPoints, distanceMeasure)]
    
    if (distanceMeasure == 'cosine'):
        return 1 - cosine_similarity(p[1], center)
    elif (distanceMeasure == 'pearson'):
        return 1 - pearson_similarity (p[1], center)
    return sqrt(sum([x**2 for x in (p[1] - center)]))



class MyKmeans():
    def __init__(self, K, tol=0.0001, maxIter=50, seed=1, distanceMeasure='cosine'):
        self.K = K
        self.tol = tol
        self.maxIter = maxIter
        self.seed = seed
        self.distanceMeasure = distanceMeasure
    
    # convert spark dataframe to rdd dataset
    def transform(self, df, featuresStartIndex=1):
        # started postion of features columns
        i = featuresStartIndex
        # parsed dataframe to rdd dataset with format: [(key, [feature cols])]
        return df.rdd.map(
                lambda r: (r[0], np.array([int(x) for x in r[i:]]))).cache()
    
    def initKPoints(self, data):
        # get feature cols from random K points
        return data.map(lambda r: r[1]).takeSample(False, self.K, self.seed)
    
    

    # train dataset and return K centroids
    def train(self, df, featuresStartIndex=1):
        tempDist = 1.0
        data = self.transform(df, featuresStartIndex)
        kPoints = self.initKPoints(data)
        convergeDist = self.tol
        distMeasure = self.distanceMeasure
        iter = 0
        
        while (tempDist > convergeDist) and (iter < self.maxIter):
            closest = data.map(
                lambda p: (closestPoint(p, kPoints, distMeasure), (p[1], 1))) # set cluster for data points
            pointStats = 0
            if (distMeasure == 'cosine'):
                pointStats = closest.reduceByKey(
                             lambda p1_c1, p2_c2: (p1_c1[0]/np.linalg.norm(p1_c1[0]) + p2_c2[0]/np.linalg.norm(p2_c2[0]),                                    p1_c1[1] + p2_c2[1]))
            elif (distMeasure == 'pearson'):
                pointStats = closest.reduceByKey(
                lambda p1_c1, p2_c2: 
               ((p1_c1[0] - np.mean(p1_c1[0]))/(np.linalg.norm(p1_c1[0] - np.mean(p1_c1[0]))) + (p2_c2[0] - np.mean(p2_c2[0]))/(np.linalg.norm(p2_c2[0] - np.mean(p2_c2[0]))), p1_c1[1] + p2_c2[1]))
            else:
                pointStats = closest.reduceByKey(
                lambda p1_c1, p2_c2: (p1_c1[0] + p2_c2[0], p1_c1[1] + p2_c2[1])) # create tuple: (sum(points),sum(count))
            newPoints = pointStats.map(
                lambda st: (st[0], st[1][0] / st[1][1])).collect() # find new centroids
    
            # compute total distances between old vs new centroids
            if (distMeasure == 'cosine'):
                tempDist = sum(1 - cosine_similarity(kPoints[iK], p) for (iK, p) in newPoints)
            elif (distMeasure == 'pearson'):
                tempDist = sum(1 - pearson_similarity(kPoints[iK], p) for (iK, p) in newPoints)
            else:
                tempDist = sum(np.sum((kPoints[iK] - p) ** 2) for (iK, p) in newPoints)

            
#             print(tempDist)
            # update new centroids
            for (iK, p) in newPoints:
                kPoints[iK] = p
                
        return kPoints
    # return new df with prediction col
    def predict(self, df, kPoints, featuresStartIndex=1):
        distMeasure = self.distanceMeasure
        data = self.transform(df, featuresStartIndex)
        # assign cluster index to each data points
        data_withLabel = data.map(
            lambda p: (p[0], closestPoint(p,kPoints, distMeasure)))
        
        result = spark.createDataFrame(data_withLabel, [df.columns[0], 'prediction'])
        
        return df.join(result, [df.columns[0]]).select(df.columns[0], 'prediction', *(col(c).cast('int').alias(c) for c in df.columns[featuresStartIndex:]))

    def summarize(self, result_df):
        count = result_df.groupBy('prediction').count().sort('prediction').rdd.map(lambda r: r[1]).collect()
        sum_count = sum(count)
        sum_set = result_df.groupBy('prediction').sum().drop('sum(prediction)').sort('prediction').rdd.map(lambda r: np.array(r[1:])).collect()
        return {'count': count, 'sum_count': sum_count, 'sum_set': sum_set}
    
    def computeCost(self, df, kPoints):
        data = self.transform(df)
        dist = self.distanceMeasure
        return data.map(lambda point: error(point, kPoints, dist)).reduce(lambda x, y: x + y)
    
    def computeEntropy(self, count, sum_count, sum_set):
        entropy_set = [entropy(s) for s in sum_set]
        
        result = 0
        for i in range(len(count)):
            result += (count[i]/sum_count)*entropy_set[i]
        
        return result
    
    def computePurity(self, count, sum_count, sum_set):
        purity_set = [np.max(s/sum(s)) for s in sum_set]
        
        result = 0
        for i in range(len(count)):
            result += (count[i]/sum_count)*purity_set[i]
            
        return result