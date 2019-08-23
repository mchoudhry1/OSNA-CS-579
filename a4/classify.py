"""
Classify data.
"""
from collections import Counter
import requests
import pickle
import re
from collections import defaultdict
from scipy.sparse import lil_matrix
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import defaultdict
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import json

def download_afinn():
    afinn = dict()
    url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
    zipfile = ZipFile(BytesIO(url.read()))
    afinn_file = zipfile.open('AFINN/AFINN-111.txt')

    for line in afinn_file:
        parts = line.strip().split()
        if len(parts) == 2:
            afinn[parts[0].decode("utf-8")] = int(parts[1])

    return afinn

def read_tweets():
    tweets = pickle.load(open('tweets.pkl', 'rb'))
    tweets = [t['text'] for t in tweets][:200]
#     tweets2 = [t for t in tweets]
    return tweets


def afinn_sentiment(terms, afinn):
    positive = 0
    negative = 0
    for t in terms:
        if t in afinn:
            if afinn[t] > 0:
                positive += afinn[t]
            else:
                negative += -1 * afinn[t]
    return positive, negative

def positive_negative(tweets, afinn):
    positives = []
    negatives = []
    
    def tokenize1(tweets):
        return re.sub('\W+', ' ', tweets.lower()).split()

    tokens = [tokenize1(t) for t in tweets]
    for token_list, tweet in zip(tokens, tweets):
        pos, neg = afinn_sentiment(token_list, afinn)
        if pos > neg:
            positives.append((tweet, pos, neg))
        elif neg > pos:
            negatives.append((tweet, pos, neg))
    
    positives = positives[:10]
    negatives = negatives[:10]
    file = open("pos_neg_tweets.txt","w", encoding = "utf-8")

    print('Here are the positive scores for all the tweets.')
    file.write('Here is the first positive tweets.\n')
    for tweet, pos, neg in sorted(positives,key=lambda x: x[1],reverse=True):
        print(pos, neg, tweet)
    file.write(positives[0][0])

    
    print('\n')
    
    print('Here are the Negative scores for all the tweets.\n')
    file.write('\n\nHere is the first Negative tweets.\n')
    for tweet, pos, neg in sorted(negatives, key=lambda x: x[2], reverse=True):
        print(pos, neg, tweet)    
    file.write(negatives[0][0])
    # return positives, negatives

def get_names():

    males1 = requests.get('https://www2.census.gov/topics/genealogy/1990surnames/dist.male.first')
    males = males1.text.split('\n')
    females1 = requests.get('https://www2.census.gov/topics/genealogy/1990surnames/dist.female.first')
    females = females1.text.split('\n')
    
    males_pact = dict([(m.split()[0].lower(), float(m.split()[1]))
                  for m in males if m])
    females_pct = dict([(f.split()[0].lower(), float(f.split()[1]))
                    for f in females if f])
    
    male_names = set([m for m in males_pact if m not in females_pct or
                  males_pact[m] > females_pct[m]])
    female_names = set([f for f in females_pct if f not in males_pact or
                  females_pct[f] > males_pact[f]])    
    return male_names, female_names



def tokens(tweet, use_descr=True, lowercase=True,
                 keep_punctuation=True, descr_prefix='d=',
                 collapse_urls=True, collapse_mentions=True):

    def tokenize(string, lowercase, keep_punctuation, prefix,
             collapse_urls, collapse_mentions):

        if not string:
            return []
        if lowercase:
            string = string.lower()
        tokens = []
        if collapse_urls:
            string = re.sub('http\S+', 'Its_A_URL', string)
        if collapse_mentions:
            string = re.sub('@\S+', 'Its_A_MENTION', string)
        if keep_punctuation:
            tokens = string.split()
        else:
            tokens = re.sub('\W+', ' ', string).split()
        if prefix:
            tokens = ['%s%s' % (prefix, t) for t in tokens]
        return tokens             

    tokens = tokenize(tweet['text'], lowercase, keep_punctuation, None,
                       collapse_urls, collapse_mentions)
    if use_descr:
        tokens.extend(tokenize(tweet['user']['description'], lowercase,
                               keep_punctuation, descr_prefix,
                               collapse_urls, collapse_mentions))
    return tokens

def vocabulary(tokens_list):
    vocabulary = defaultdict(lambda: len(vocabulary))  
    for tokens in tokens_list:
        for token in tokens:
            vocabulary[token]  
    print('%d Number of unique terms in vocabulary' % len(vocabulary))
    return vocabulary

def csr_matrix(tokens_list, vocabulary,tweets):
    X = lil_matrix((len(tweets), len(vocabulary)))
    for i, tokens in enumerate(tokens_list):
        for token in tokens:
            j = vocabulary[token]
            X[i,j] += 1
    return X.tocsr()


def getting_gender(tweet, male_names, female_names):
    
    def get_first_name(tweet):
        if 'user' in tweet and 'name' in tweet['user']:
            parts = tweet['user']['name'].split()
        if len(parts) > 0:
            return parts[0].lower()

    name = get_first_name(tweet)
    if name in female_names:
        return 1
    elif name in male_names:
        return 0
    else:
        return -1

def cross_validation_accuracy(X, y, nfolds):
    
    cv = KFold(n_splits=nfolds,random_state=42, shuffle=True)
    accuracies = []
    for train_idx, test_idx in cv.split(X):
        clf = LogisticRegression()
        clf.fit(X[train_idx], y[train_idx])
        predicted = clf.predict(X[test_idx])
        acc = accuracy_score(y[test_idx], predicted)
        accuracies.append(acc)
    average = np.mean(accuracies)
    # print(accuracies)
    return average


def main():
    print("Sentiment analysis based on AFINN...")
    afinn = download_afinn()
    tweets = read_tweets()
    print('Found %d Tweets' % (len(tweets)))
    positive_negative(tweets, afinn)
    print('\n')
    
    positive_negative(tweets, afinn)
    
    print('Lets do Gender Classfication')
    tweets1 = pickle.load(open('tweets.pkl', 'rb'))
    tweets2 = [t for t in tweets1]
#     print (tweet)

    male_names, female_names = get_names()
    print('Number of Males: %.0f, Number of Females: %.0f' %(len(male_names), len(female_names)))
    print('\n')
    tokens_list = [tokens(t, use_descr=True, lowercase=True,
                            keep_punctuation=True, descr_prefix='d=',
                            collapse_urls=True, collapse_mentions=False)
              for t in tweets2]

    vocabulary1 = vocabulary(tokens_list)

    X = csr_matrix(tokens_list, vocabulary1,tweets2)
    print('shape of X:', X.shape)
    y = np.array([getting_gender(t, male_names, female_names) for t in tweets2])
    file = open('gender_labels.txt', 'w')
    file.write(str(Counter(y)))
    print('gender labels:', Counter(y))
    print('avg accuracy', cross_validation_accuracy(X, y, 5))    
    
    Answers = cross_validation_accuracy(X, y, 5)
    pickle.dump(Answers, open('Answers.pkl','wb'))

if __name__ == "__main__":
    main()