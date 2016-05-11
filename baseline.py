import sys
reload(sys)
sys.setdefaultencoding("ISO-8859-1")
import re
from nltk.corpus import opinion_lexicon
#import nltk.corpus
flag = 0
sentiment_words = {}

#set up the lexicon
for w in nltk.corpus.opinion_lexicon.words():
    if flag == 0:
        sentiment_words[w] = flag
        if re.search("zombie",w):
            flag = 1
    else:
        sentiment_words[w] = flag

test_file = open("test.list",'rb')
out_file = open("baseline_out.txt",'w')
line_cnt = 1
for tweet in test_file:
    words = tweet.split(" ")
    neg = 0
    pos = 0
    for word in words:
        try:
            if sentiment_words[word.lower()] == 1:
                pos += 1
            elif sentiment_words[word.lower()] == 0:
                neg += 1
        except KeyError:
            pass

    if pos > neg:
        out_file.write(str(line_cnt) + " positive\n")
    elif pos < neg:
        out_file.write(str(line_cnt) + " negative\n")
    else:
        out_file.write(str(line_cnt) + " neutral\n")

    line_cnt += 1


 
