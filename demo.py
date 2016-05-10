import knn
import naiveBayes

trainingSet,lexicon = knn.train()
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
        print("Baseline: positive")
    elif pos < neg:
        print("Baseline: negative")
    else:
        print("Baseline: neutral")
    
    #Naive Bayes 
    naiveBayes.trainNaiveBayes("train_formatted.txt")
    print "Naive Bayes: "+ naiveBayes.makePrediction(test_in)

    #KNN
    knn_test_feats = knn.processTest(lexicon,test_in)
    knn_pred = knn.test(trainingSet, knn_test_feats, k)
    print("KNN (k=5): " + knn_pred)

