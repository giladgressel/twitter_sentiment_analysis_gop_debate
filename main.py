import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(color_codes=True)

from preprocessing.load_data import load_gop_data

df_gop = load_gop_data('../data/gop_Sentiment.csv')

#let's look for what kinds of unique values we have in our sentiment scoring category.
# We should also check if we have NaN's that we'd want to deal with
print df_gop.sentiment.unique()
print ""
print "Columns have null values?"
print df_gop.isnull().any()

#we will want our labels numeric (some classifiers may not like 3-way text labels),
#we can do that all now.

df_gop.sentiment.replace(['Positive', 'Negative', 'Neutral'], [1, -1 , 0], inplace = True)
print "gop tweets \n"
print df_gop[:10]

#Let's take a look at our class distribution
total_tweets = len(df_gop)
positive_tweets = sum(df_gop.sentiment == 1)
negative_tweets = sum(df_gop.sentiment == -1)
neutral_tweets = sum(df_gop.sentiment == 0)

print "The total number of samples is : {}".format(len(df_gop.sentiment))
print "There are {} positive tweets or {}%".format \
(positive_tweets, positive_tweets/float(total_tweets) )
print "There are {} Negative tweets or {}%".format \
(negative_tweets, negative_tweets / float(total_tweets))
print "There are {} Neutral tweets or {}%".format \
(neutral_tweets, neutral_tweets/ float(total_tweets))

sns.countplot(x ='sentiment', data = df_gop)

# ok let's parse some sample the tweets use the ARK tokenizer from CMU.
from twitter_sentiment_analysis_gop_debate.preprocessing import twokenize as tw

first_tweet = df_gop.iloc[0, 1]
second_tweet = df_gop.iloc[1,1]
third_tweet = df_gop.iloc[2,1]

test_tweets = [first_tweet, second_tweet, third_tweet]
for tweet in test_tweets:
    print tweet

print ""

for tweet in test_tweets:
    print tw.tokenizeRawTweetText(tweet)


#stop-word removal
with open('stopwords.txt') as f:
    stop_words = f.read().splitlines()
    stop_words.extend(['I', r'\.', 'The',r'\.\.']) #add upper-case I, and escaped periods
print stop_words

# compile a regex for stop words, urls, rt's, etc

retweets = re.compile(r'(rt ?@.*?:)')
urls = re.compile(r'(https?:.*\b)')
stop_word = re.compile(r'\b(?:{})\b'.format('|'.join(stop_words)))

regex_args = [retweets, urls, stop_word]
#make a method to easily use our regex : note we do not want to compile the regex within this def, that would be computationally expensive.
def parse_tweet (tweet , retweets, urls, stop_word):
    tweet = tweet.lower()
    tweet = re.sub(retweets, "", tweet) #removes RT@thisguy: or RT @thisguy:   two common Retweet bits I dont' need
    tweet = re.sub(urls, "", tweet) # removes URL's
    tweet = re.sub(stop_word, "", tweet) #removes stop words.
    tweet = tw.tokenizeRawTweetText(tweet)
    if len(tweet) < 1:
        return "NaN"
    else:
        return tweet

for tweet in test_tweets:
    print parse_tweet(tweet, *regex_args)


#parse all tweets in dataframe
# ok, now that we have rough parsing, lets parse them all!
#run a decoder on unicode, cause
df_gop.text = df_gop.text.apply(lambda x: x.decode('utf-8'))
df_gop.text = df_gop.text.apply(lambda x: parse_tweet(x,*regex_args))
print df_gop.head()
print df_gop.shape


# let's check to make sure we didnt' parse anything down to an empty string or became NaN
print df_gop.isnull().any()
print np.where(df_gop.applymap(lambda x: x == ''))


# now that we have all the tweets parsed, we actually want to split into our training / testing sets.
# This is because n-gram analysis (which comes next), should not be done on the testing data!
# The n-gram analysis should on be on training data.

# TODO try to implement n-gram analysis with cross validation, for now I'll use a hold-out testing set
from sklearn import cross_validation

#Let's split up the labels from the training data

X_all = df_gop['text']
y_all = df_gop['sentiment']

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X_all, y_all, test_size=0.25, stratify = y_all, random_state = 0)  #note we run stratify, in order to try and keep a balanced labeled set

print "size of training tweets: ", len(X_train)
print "size of testing tweets: ", len(X_test)

plt.figure(figsize = (15,6))
plt.subplot(1,2,1)
ax = sns.countplot(y_train)
ax.set_title("Training Labels")
plt.subplot(1,2,2)
ax = sns.countplot(y_test)
ax.set_title("Testing Labels")
plt.show


# we need to merge the labels and parsed tweets for doing our n-gram analysis.
# This is because we will build n-gram models for each class, therefore we need to select only
# those tweets that are positive / negative for the two n-gram tables.

# Let's re-merge the labels into the training data order to do n-gram analysis
XyN_gram = pd.concat([X_train, y_train], axis = 1)

print XyN_gram.iloc[0, 0]
print XyN_gram.head()


#Let's start by developing a function that will take a parsed tweet and output grams of any size

test1 = XyN_gram.iloc[0,0]

uni_gram_map = {}
bi_gram_map = {}
tri_gram_map = {}

def nGram_counter (parsed_tweet, distance_to_cover, gram_map):
    for_loop_range = range(len(parsed_tweet) - distance_to_cover)
    for i in for_loop_range:
        gram = tuple(parsed_tweet[i:i+distance_to_cover])
        if gram in gram_map:
            gram_map[gram] += 1
        else:
            gram_map[gram] = 1

nGram_counter(test1, 3, tri_gram_map)
nGram_counter(test1, 2, bi_gram_map)
nGram_counter(test1, 1, uni_gram_map)

# check our output.
print "the test tweet: {}".format(test1)
print ""
print uni_gram_map
print " "
print bi_gram_map
print " "
print tri_gram_map
print " "


#let's now apply our n-gram counter to all the tweets of a certain class.
# Let's make the netural n_gram map, on the training data.

#setup n-gram maps.
neutral_uni_gram_map ={}
neutral_bi_gram_map = {}
neutral_tri_gram_map = {}

neutral_tweets = XyN_gram[XyN_gram.sentiment == 0]

neutral_tweets.apply(lambda x: nGram_counter(x.text, 1, neutral_uni_gram_map), 1)
neutral_tweets.apply(lambda x: nGram_counter(x.text, 2, neutral_bi_gram_map), 1)
neutral_tweets.apply(lambda x: nGram_counter(x.text, 3, neutral_tri_gram_map), 1)
print "Total Unigrams for Neutral Tweets : {}".format(len(neutral_uni_gram_map))
print
print "Total Bi-grams for Neutral Tweets: {}".format(len(neutral_bi_gram_map))
print
print "Total Tri-grams for Neutral Tweets: {}".format(len(neutral_tri_gram_map))
print
print "Most popular Neutral Uni-grams : {}" \
.format(sorted(neutral_uni_gram_map.items(), key=lambda x: x[1], reverse = True)[:30])
print
print "Most Popular Neutral  Bi-gams : {}" \
.format(sorted(neutral_bi_gram_map.items(), key = lambda x: x[1], reverse = True)[:30])
print
print "Most popular Neutral Tri-grams : {}" \
.format(sorted(neutral_tri_gram_map.items(), key=lambda x: x[1], reverse = True)[:30])


# maximum likliehood probabilities for neutral grams.
# we will calculate maximum likliehood with smoothing, will use simple laplace k-smoothing, with k = 1


def calculate_maximum_likliehood (gram_map, k_smoothing = 1, Prior_map = None):
    MLE_estimates = {}
    count_vocabulary = len(gram_map) # this is V, or the unique vocabulary
    total_gram_count = sum(gram_map.values()) #the is the total count of all the grams

    if Prior_map != None:
        prior_gram_vocabulary_count = len(Prior_map) # also V for smoothing on conditioned grams

    #figure out what kind of gram-map we have
    keys = gram_map.keys()
    if len(keys[0]) == 1: # we have unigrams
        for key in keys:
            MLE_estimates[key] = (gram_map[key] + k_smoothing) / \
            float((count_vocabulary*k_smoothing) + total_gram_count)
            # above will give MLE with smoothing = 1

    elif len(keys[0]) == 2: # We have bigrams, thus we sould condition on a previous unigram
        for key in keys:
            MLE_estimates[key] = (gram_map[key] + k_smoothing) / \
            float((k_smoothing * prior_gram_vocabulary_count) + Prior_map[key[0],])
    elif len(keys[0]) == 3: #should be 3 size, so condition on previous bi-gram
        for key in keys:
            MLE_estimates[key] = (gram_map[key] + k_smoothing) / \
            float((k_smoothing * prior_gram_vocabulary_count) +Prior_map[key[:2]])
    else: #should never get here
        print "whoa, what are you passing?"
        print key


    return MLE_estimates

MLE_neutral_uni_gram = calculate_maximum_likliehood(neutral_uni_gram_map , 1)
MLE_neutral_bi_gram = calculate_maximum_likliehood(neutral_bi_gram_map, 1, Prior_map=neutral_uni_gram_map)
MLE_neutral_tri_gram = calculate_maximum_likliehood(neutral_tri_gram_map, 1, Prior_map=neutral_bi_gram_map)

## sanity checks
print len(MLE_neutral_uni_gram) == len(neutral_uni_gram_map)
print len(MLE_neutral_bi_gram) == len(neutral_bi_gram_map)
print len(MLE_neutral_tri_gram) == len(neutral_tri_gram_map)

# should look reasonble?
print MLE_neutral_bi_gram.values()[:10]


import math

def v_plus_n(grams):
    total_unique_grams = len(grams) # this is V or the unique vocabulary for smoothing
    total_gram_count = sum(grams.values()) #this is N, total number of counts of the vocabulary.
    return float(total_unique_grams + total_gram_count)


def probability_calculator (parsed_tweet, gram_size):

    if len(parsed_tweet) <1:  #this will catch any empty tweets I missed earlier.
        print "you have a NAN"
        print parsed_tweet
        return "NaN"

    uni_VplusN = v_plus_n(neutral_uni_gram_map) # will use these values in smoothing
    bi_VplusN = v_plus_n(neutral_bi_gram_map)
    tri_VplusN = v_plus_n(neutral_tri_gram_map)

    # gram_map should correspond to gram_size i.E bi-grams, or tri-grams etc.
    loop_range = range(len(parsed_tweet) - gram_size)
    prob = 0

    if gram_size == 1: #unigrams
        for i in loop_range:
            gram = tuple(parsed_tweet[i:i+gram_size])

            if gram in MLE_neutral_uni_gram: #look up the probability value we've already calculated
                prob += math.log(MLE_neutral_uni_gram[gram])
            else:  #it's unseen so create a new probability with k-smoothing
                #pass # penalize it with nothing
                prob += math.log( 1.0 / uni_VplusN )

    if gram_size == 2: #bi-grams
        for i in loop_range:
            gram = tuple(parsed_tweet[i:i+gram_size])

            if gram in MLE_neutral_bi_gram:
                prob += math.log(MLE_neutral_bi_gram[gram])  #look up probability we've calculated

            else:  #condition the unseen bi-gram on the seen unigram.
                #pass
                if (gram[0],) in neutral_uni_gram_map:
                    prob += math.log( 1.0 / (neutral_uni_gram_map[gram[0],] + len(neutral_uni_gram_map)))

                    # so if gram = ('this','cat'), and we have never seen that before.  we are
                    # getting a probability that is: 1 / count('this') + count(unique_single grams)
                    #obviously close to zero.  ....
                else: #then if even the first part of this unseen bigram is not the unigram database, just do V+N
                    prob += math.log(1.0 / bi_VplusN)

    if gram_size == 3: #tri-grams
        for i in loop_range:
            gram = tuple(parsed_tweet[i:i+gram_size])

            if gram in MLE_neutral_tri_gram:
                prob += math.log(MLE_neutral_tri_gram[gram]) # look up prob we've already calculated

            else:
                #pass
                if gram[:2] in neutral_bi_gram_map:
                    prob += math.log( 1.0 / (neutral_bi_gram_map[gram[:2]] + len(neutral_bi_gram_map)))
                else:
                    prob += math.log(1.0 / tri_VplusN)

    probability = math.exp(prob) / len(parsed_tweet) # normalize by the number of grams in the tweet.
    return probability


test_tweet = [u'ok', u',', u'can', u'cull', u'herd', u'two', u'#gopdebates', u'?', u'can', u'tell', u'pataki', u',', u'carson', u',', u'walker', u',', u'cruz', u'perry', u'go', u'home', u'?']
print probability_calculator(test_tweet,1)
print probability_calculator(test_tweet,2)
print probability_calculator(test_tweet,3)


X_trainy = pd.DataFrame(X_train)  #have to convert the Series into a dataframe, in order to add columns
X_trainy['neut-uni'] = X_trainy.text.apply(lambda x: probability_calculator(x, 1),1)
X_trainy['neut-bi'] = X_trainy.text.apply(lambda x: probability_calculator(x,2),1)
X_trainy['neut-tri'] = X_trainy.text.apply(lambda x: probability_calculator(x,3),1)
print X_trainy.head()

#note that X_test is getting probability, but they are calculated based on X_train's probability model

X_testy = pd.DataFrame(X_test) #have to convert the Series into a dataframe, in order to add columns
X_testy['neut-uni'] = X_testy.text.apply(lambda x: probability_calculator(x, 1), 1)
X_testy['neut-bi'] = X_testy.text.apply(lambda x: probability_calculator(x,2),1)
X_testy['neut-tri'] = X_testy.text.apply(lambda x: probability_calculator(x,3),1)
print X_testy.head()


#First let's drop the text tweets, they aren't helpful in actual classification
X_train_go = X_trainy.drop(X_trainy.columns[0], axis =1)
print X_train_go.head()
X_test_go = X_testy.drop(X_testy.columns[0], axis =1)
print X_test_go.head()

#should be no reason to scale data, because we've normalized it all, it's all probabilities.

from sklearn.metrics import f1_score
from sklearn import svm
from sklearn import tree
from sklearn.metrics import confusion_matrix



def f1_score_wrap (y_actual, y_predict):
    return f1_score(y_actual, y_predict, average = "weighted")


clf = tree.DecisionTreeClassifier()
def basic(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)

    x_pred = clf.predict(X_train)
    F1_train = f1_score_wrap(y_train, x_pred)
    train_conf = confusion_matrix(y_train, x_pred)

    print "training F1:", F1_train
    print
    print "training confusion:\n", train_conf
    print

    y_pred = clf.predict(X_test)
    F1_score = f1_score_wrap(y_test, y_pred)
    conf = confusion_matrix(y_test, y_pred)

    print "testing F1:", F1_score
    print
    print "confusion for testing\n", conf

basic(clf, X_train_go, X_test_go, y_train, y_test)
print X_test_go.shape
print X_train_go.shape


SVM = svm.SVC()
basic(SVM, X_train_go, X_test_go, y_train, y_test)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
basic(lr, X_train_go, X_test_go, y_train, y_test)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
basic(gnb, X_train_go, X_test_go, y_train, y_test)

from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators=100)
basic(ada, X_train_go, X_test_go, y_train, y_test)
