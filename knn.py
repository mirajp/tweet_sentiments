import sys
reload(sys)
sys.setdefaultencoding("ISO-8859-1")
import re
import math
import operator
from nltk.corpus import opinion_lexicon

def main():
    k = 5
    trainingSet,lex = train()
    testFile = open("test.list",'rb')
    outFile = open("knn_out.txt","w")
    line_cnt = 1
    for line in testFile:
        feats = processTest(lex,line)
        #print feats
        pred = test(trainingSet, feats, k)
        outFile.write(str(line_cnt) + " " + pred+ "\n")
        line_cnt += 1
            

def processTest(lexicon,testPhrase):
    a = testPhrase.split(" ")
    feats = []
    pos = 0
    neg = 0
    chars = 0
    for word in a:
        chars += len(word)
        try:
            if lexicon[word.lower()] == 1:
                pos += 1
            elif lexicon[word.lower()] == 0:
                neg += 1
        except KeyError:
            pass  

    feats.append(pos)
    feats.append(neg)
    feats.append(chars)

    return feats


def test(trainingSet,testInstance, k):
    neighbors = getNeighbors(trainingSet,testInstance,k)
    pred = getResponse(neighbors)
    return pred

def train():
    lexicon = prepLexicon()
    trainingSet = []
    trainFile = open("train_formatted.txt","rb")
    for line in trainFile:
        features = []
        pos = 0
        neg = 0
        chars = 0
        parts = line.split("\t")
        label = parts[0].replace('"',"")
        if re.search("objective",label) or re.search("objective-OR-neutral",label):
            label = "neutral"
        for word in parts[1].split(" "):
            chars += len(word)
            try:
                if lexicon[word.lower()] == 1:
                    pos += 1
                elif lexicon[word.lower()] == 0:
                    neg += 1
            except KeyError:
                pass
        
        features.append(pos)
        features.append(neg)
        features.append(chars)
        features.append(label)
        trainingSet.append(features)

    return trainingSet,lexicon
        
                
            
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
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):        
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1),reverse=True)
    return sortedVotes[0][0]

if __name__ == "__main__":
    main()
