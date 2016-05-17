import knn
import naiveBayes
import sys
import os

trainingSet,lexicon = knn.train()
naiveBayes.trainNaiveBayes("train_formatted.txt")
k = 5
while(1):
    test_in = raw_input("Input: ")
    if test_in == "q":
        exit()    
    
    #baseline
    pos = 0
    neg = 0    
    words = test_in.split(" ")
    for word in words:
        try:
            if lexicon[word.lower()] == 1:
                pos += 1
            elif lexicon[word.lower()] ==0:
                neg += 1
        except KeyError:
            pass
    
    if pos > neg:
       pass
       #print("Baseline: positive")
    elif pos < neg:
        pass
        #print("Baseline: negative")
    else:
        pass
        #print("Baseline: neutral")
   
    #Naive Bayes 
    print "Naive Bayes: "+ naiveBayes.makePrediction(test_in)
    if naiveBayes.makePrediction(test_in) == "positive":
        os.system("sudo ../rpi-rgb-led-matrix/led-matrix -r 16 -t 2 -D 0")
    elif naiveBayes.makePrediction(test_in) == "neutral":
        os.system("sudo ../rpi-rgb-led-matrix/led-matrix -r 16 -t 2 -D 5")
    else:
	os.system("sudo ../rpi-rgb-led-matrix/led-matrix -r 16 -t 2 -D 4")
    #KNN
 #   knn_test_feats = knn.processTest(lexicon,test_in)
 #   knn_pred = knn.test(trainingSet, knn_test_feats, k)
 #   print("KNN (k=5): " + knn_pred)

