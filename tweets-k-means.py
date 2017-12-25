import numpy as np
import sys
import json
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import re
import string

# Jaccard distance between the tweets
def getDistance(tweet1, tweet2):
    global tweetsDictionary
    union = len(tweetsDictionary[tweet1] | tweetsDictionary[tweet2] )
    intersection = len(tweetsDictionary[tweet1] & tweetsDictionary[tweet2])
    return 1 - np.divide(intersection,union)

# Assign the tweets to different clusters
def assign(centers):
    global tweetsDictionary, k
    clusters = [[] for i in range(k)]
    # Check for all data points
    for key in tweetsDictionary:        
        distance = []
        # Check distance with all centers for a point
        for i in range(len(centers)):
            dist = getDistance(key,int(centers[i]))
            distance.append(dist)      
        # print(distance)
        # print(np.argmin(distance))
        clusters[np.argmin(distance)].append(key)
    return clusters

# Update the centers of each cluster
def updateCenters(clusters,centers):
    global tweetsDictionary
    # For all the clusters, calculate the centroid
    for i in range(len(clusters)):
        clusterLength = len(clusters[i])

        interDistanceList = []
        for x in range(clusterLength):
            minDistance = 0.0
            for y in range(clusterLength):
                minDistance += getDistance(clusters[i][x],clusters[i][y])
            interDistanceList.append(minDistance)         
        centers[i] = clusters[i][np.argmin(interDistanceList)]
    return centers      

# Calculate SSE
def calculateSSE(clusters,centers):
    global k, tweetsDictionary
    dist = 0.0
    # For every cluster
    for i in range(k):
        # Every point in a cluster
        for x in range(len(clusters[i])):
            tweetId = clusters[i][x]
            dist += np.square(getDistance(centers[i],tweetId))  
    # print("SSE: ",dist)
    return dist

def kmeans(centers):
    global tweetsDictionary, outputFile
    
    while(True):
        clusters = assign(centers)
        oldCenter = list(centers)     
        centers = updateCenters(clusters,centers)
        if(oldCenter == centers):
            break
    
    # print("Clusters")
    # for i in range(len(clusters)):
    #     print(i+1,'\t',','.join(str(x) for x in list(clusters[i])))
    print("SSE: ",calculateSSE(clusters,centers))

    with open(outputFile, mode='w') as output:
        for i in range(len(clusters)):
            # output.write(str(i+1)+'\t'+','.join(str(x) for x in list(clusters[i]))+'\n')
            output.write(str(i+1)+'\t'+",".join([str(s) for s in clusters[i]])+'\n')

        output.write("\nSSE: " + str(calculateSSE(clusters,centers)))

def getCleanedTokensSet(sentence):
    global table, tokenizer
    sentence = sentence.lower()
    sentence = re.sub('[\n\r\t]',' ',sentence)
    sentence = re.sub('@\w+:*','',sentence)
    sentence = re.sub('(^rt|\\bvia\\b)','',sentence)
    sentence = re.sub('http\S+','',sentence) 
    sentence = sentence.translate(table)
    tokens = tokenizer.tokenize(sentence)
    final = [w for w in tokens if not w in stopwords.words('english') and len(w) > 2]
    return set(final)
    

if __name__ == '__main__':

    # Check for input arguments
    if(len(sys.argv)<5):
        print("Please enter all the parameters required. E.g. python tweets-k-means.py <numberOfClusters> <initialSeedsFile> <TweetsDataFile> <outputFile>")
        exit()

    k = int(sys.argv[1])
    seedsFile = '.\\' + sys.argv[2]
    inputFile = '.\\' + sys.argv[3]
    outputFile = '.\\' + sys.argv[4]

    
    tokenizer = RegexpTokenizer(r'\w+')
    table = str.maketrans({key: None for key in string.punctuation})

    tweetsDictionary = {}
    centers =[]

    # Initializing centers of the clusters using the input seeds
    with open(seedsFile) as seeds:
        for line in seeds:
            centers.append(line.rstrip(',\n'))

    # print("Centers: ",centers)
    
    # Creating tweets dicionary using the input tweets in json format
    with open(inputFile, mode='rb') as jsonFile:
        for jsonObject in jsonFile:
            datastore = json.loads(jsonObject)
            tweetsDictionary[datastore['id']] = getCleanedTokensSet(datastore['text'])       

    # print(tweetsDictionary)
    # print("Datapoints: ",dataPoints)
    kmeans(centers)
    
    