import sys
reload(sys)
sys.setdefaultencoding("ISO-8859-1")
import re
import math
import operator
from nltk.corpus import opinion_lexicon

def main():
    train()

def train():
    lexicon = prepLexicon()
    trainingSet = []
    features = []
    trainFile = open("train_formatted.txt","rb")
    for line in trainFile:
        pos = 0
        neg = 0
        chars = 0
        parts = line.split("\t")
        label = parts[0].replace('"',"")
        for word in parts[1].split(" "):
            chars += len(word)
            
            

def prepLexicon():
    sentiment_words = {}
    flag = 0
    for w in opinion_lexicon.words():
        if flag == 0:
            sentiment_words[w] = flag
            if re.search("zombie",w):
                flag = 1
        else:
            sentiment_words[w] = flag
    return sentiment_words

def euclideanDistance(instance1,instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]),2)
    return math.sqrt(distance)

def getNeighbors(trainingSet,testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x],dist))
        distances.sort(key=operator.itemmgetter(1))
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])
        return neighbors

def test(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritemms(), key=operator.itemgetter(1),reverse=True)
    return sortedVotes[0][0]

if __name__ == "__main__":
    main()
