import sys  
reload(sys)  
sys.setdefaultencoding("ISO-8859-1")

from nltk.stem import WordNetLemmatizer
from nltk.corpus import opinion_lexicon
from nltk.corpus import wordnet
import re
from nltk.corpus import stopwords
from nltk.corpus import words as words
from math import log
from nltk import word_tokenize, pos_tag

demomode = 0
if demomode == 1:
    print "Running in demo mode..."
wordnet_lemmatizer = WordNetLemmatizer()
engstopwords = stopwords.words("english")
engwords = words.words()
mystopwords = {}
mywords = {}
sentiment_words = {}

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('PR'):
        return wordnet.NOUN
    elif treebank_tag.startswith('RB'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# sentimentBoostVal = how much to increase word's count in category if it correctly identifies tweet's category
sentimentBoostVal = 20
# alpha = additive smoothing factor
alpha = 5

#Count number of times each word appears in each category
wordsCount = {}
#Count the total number of words in each category
totalCount = {}
#Keep track of vocabulary across entire training
vocab = {}
vocabSize = 0
#Count the number of documents in each category
tweetCount = {}
numTweets = 0.0
testProbs = {}


for i in range(0, len(engstopwords)):
    mystopwords[engstopwords[i]] = 1

for i in range(0, len(engwords)):
    mywords[engwords[i]] = 1

flag = 0
#0 = negative, 1 = positive
for w in opinion_lexicon.words():
    if flag == 0:
        sentiment_words[w] = flag
        if re.search("zombie",w):
            flag = 1

    else:
        sentiment_words[w] = flag

sentiment_words[":)"] = 1
sentiment_words[":("] = 0
sentiment_words["(:"] = 1
sentiment_words["):"] = 0
sentiment_words["=)"] = 1
sentiment_words["(="] = 1
sentiment_words["=("] = 0
sentiment_words[")="] = 0
sentiment_words[":-)"] = 1
sentiment_words["(-:"] = 1
sentiment_words[":-("] = 0
sentiment_words[")-:"] = 0
sentiment_words["=-("] = 0
sentiment_words["=-)"] = 1

# less popular smileys?
sentiment_words[":')"] = 1
sentiment_words[":-')"] = 1
sentiment_words["=')"] = 1
sentiment_words["=-')"] = 1
sentiment_words[":'("] = 0
sentiment_words[":-'("] = 0
sentiment_words["=-'("] = 0
sentiment_words["='("] = 0
sentiment_words[";)"] = 1
sentiment_words[";-)"] = 1
sentiment_words[";("] = 0
sentiment_words[";-("] = 0


def clean_word(input_word):
    cleaned_word = input_word.lower()
    cleaned_word = cleaned_word.replace("!", "")
    cleaned_word = cleaned_word.replace("?", "")
    
    #Remove last period, but not the period in initialization token
    if cleaned_word.find(".") == len(cleaned_word)-1:
        cleaned_word = cleaned_word[:-1]
    
    if (cleaned_word not in sentiment_words) and (cleaned_word in mystopwords or cleaned_word not in mywords):
        cleaned_word = ""
    
    return cleaned_word

def trainWithTweet(tweet, category):
    global alpha, vocabSize, numTweets, sentimentBoostVal
    words = word_tokenize(tweet)
    wordsTags = pos_tag(words)
    numWords = len(words)
    wordIter = 0
    while wordIter < numWords:
        foundWord = clean_word(words[wordIter])
        tag = wordsTags[wordIter][1]
        foundWord = wordnet_lemmatizer.lemmatize(foundWord, get_wordnet_pos(tag))
        
        #if len(foundWord) > 0 and foundWord not in mystopwords:
        if len(foundWord) > 0:
            totalCount[category] += 1
            if foundWord not in vocab:
                vocab[foundWord] = 1
                vocabSize += 1
            
            if foundWord not in wordsCount[category]:
                wordsCount[category][foundWord] = alpha + 1
            else:
                wordsCount[category][foundWord] += 1
            
            if foundWord in sentiment_words:
                if category == "positive" and sentiment_words[foundWord] == 1:
                    #Increase weight by boosting count
                    wordsCount[category][foundWord] += sentimentBoostVal
                    totalCount[category] += sentimentBoostVal-1
                elif category == "negative" and sentiment_words[foundWord] == 0:
                    #Increase weight by boosting count
                    wordsCount[category][foundWord] += sentimentBoostVal
                    totalCount[category] += sentimentBoostVal-1
            
        wordIter += 1

    tweetCount[category] += 1
    numTweets += 1
    return

def trainNaiveBayes(trainingList):
    linenum = 0
    trainingListFile = open(trainingList, "rb")
    for line in trainingListFile:
        line = line.split("\t")
        trainingCat = line[0].replace("\"", "")
        if trainingCat.find("objective") != -1:
            trainingCat = "neutral"
        trainingTweet = line[1]
        if trainingCat not in wordsCount:
            #print "Adding dictionary for category:", trainingCat
            wordsCount[trainingCat] = {}
            totalCount[trainingCat] = 0.0
            testProbs[trainingCat] = 0.0
            tweetCount[trainingCat] = 0.0
            
        trainWithTweet(trainingTweet, trainingCat)

    trainingListFile.close()
    return

def makePrediction(tweet):
    global alpha, vocabSize, numTweets
    prediction = ""
    for category in testProbs:
        testProbs[category] = 0

    words = word_tokenize(tweet)
    wordsTags = pos_tag(words)
    numWords = len(words)
    wordIter = 0
    while wordIter < numWords:
        foundWord = clean_word(words[wordIter])
        tag = wordsTags[wordIter][1]
        foundWord = wordnet_lemmatizer.lemmatize(foundWord, get_wordnet_pos(tag))
        
        #if len(foundWord) > 0 and foundWord not in mystopwords:
        if len(foundWord) > 0:
            for category in testProbs:
                if foundWord in wordsCount[category]:
                    testProbs[category] += log((wordsCount[category][foundWord]/(totalCount[category] + alpha*vocabSize)))
                else:
                    testProbs[category] += log((alpha/(totalCount[category] + alpha*vocabSize)))
                        
        wordIter += 1

    for category in testProbs:
        testProbs[category] += log((tweetCount[category]/numTweets))
        
    maxProb = float("-inf")
    for category in testProbs:
        #print "category '", category, "' has prob", testProbs[category]
        if testProbs[category] > maxProb:
            #print "Switching prediction to", category
            prediction = category
            maxProb = testProbs[category]

    return prediction

def testNaiveBayes(testingList, predictionsList):
    global alpha, vocabSize, numTweets
    testingListFile = open(testingList, "rb")
    predictionsListFile = open(predictionsList, "wb")
    linenum = 0
    for tweet in testingListFile:
        linenum += 1
        likelyCategory = makePrediction(tweet)
        predictionsListFile.write(str(linenum) + " " + likelyCategory + "\n")

    testingListFile.close()
    predictionsListFile.close()
    return

def main():
    trainingList = raw_input("Enter the filename of the list of training documents: ")
    if demomode == 0:
        testingList = raw_input("Enter the filename of the list of testing documents: ")
        predictionsList = raw_input("Enter the filename to save the predictions: ")

    trainNaiveBayes(trainingList)
    if demomode == 0:
        testNaiveBayes(testingList, predictionsList)
    else:
        print "Live demo mode started..."
        while True:
            print "Enter a tweet:"
            lineread = sys.stdin.readline()
            lineread = lineread[:-1]
            if lineread == '':
                break
            print "Predicted sentiment: " + makePrediction(lineread) + "\n"

    exit()

if __name__ == "__main__":
    main()

