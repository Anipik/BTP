from itertools import chain
import numpy
import string
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()
var = 0
var2 = 0
#Add features here for each word

#Contexual Features
def normalizeWord(w):
    w = w.lower()
    nw=""
    for c in w:
        if c.isdigit():
            nw+='9'
        else:
            nw+=c
    return nw
    return l

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-4:]=' + word[-4:],           #last 4 chars
        'word[-3:]=' + word[-3:],           #last 3 chars
        'word[-2:]=' + word[-2:],           #last 2 chars
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'word.lemmatized=%s' % wnl.lemmatize(word), #Lemmatized
        'postag=' + postag,                 #POS
        'postag[:2]=' + postag[:2],
        'word[:2]=' + word[:2],             # Prefix 2
        'word[:3]=' + word[:3],             # Prefix 3
        'word[:4]=' + word[:4],             # Prefix 4
        'InitCap=%s' % word[0].upper(),     #InitCap
        'AllCap=%s' % word.isupper(),       #AllCap
        'MixCase=%s' % (not (word.isupper() or word.islower())),    #MixCase
        'SingLow=%s' % (len(word)==1 and word.islower()),       #SingLow
        'SingUp=%s' % (len(word)==1 and word.isupper()),        #SingUp
        'Num=%s' % word.isnumeric(),                            #Num
        'PuncChar=%s' % (len(word)==1 and word in string.punctuation),  #PuncChar
    ]

    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1,
            '-1:postag[:2]=' + postag1[:2],
            '-1:PrevCharAN=%s' % word1[len(word1)-1].isalnum(), #PrevCharAN
        ])
    else:
        features.append('BOS')

    if len(sent) >= 2: #Then only bigram features exist
        if i > 1 :
            features.extend([
                'bigram[-2]=%s' % (sent[i-2][0] + " " + sent[i-1][0]),
                'lemBigram[-2]=%s' % (wnl.lemmatize(sent[i-2][0]) + " " + wnl.lemmatize(sent[i-1][0])),
                'normBigram[-2]=%s' % (normalizeWord(sent[i-2][0]) + " " + normalizeWord(sent[i-1][0])),
            ])
        if i > 0 :
            features.extend([
                'bigram[-1]=%s' % (sent[i-1][0] + " " + sent[i][0]),
                'lemBigram[-1]=%s' % (wnl.lemmatize(sent[i-1][0]) + " " + wnl.lemmatize(sent[i][0])),
                'normBigram[-1]=%s' % (normalizeWord(sent[i-1][0]) + " " + normalizeWord(sent[i][0])),
            ])
        if i+1 < len(sent) :
            features.extend([
                'bigram[0]=%s' % (sent[i][0] + " " + sent[i+1][0]),
                'lemBigram[0]=%s' % (wnl.lemmatize(sent[i][0]) + " " + wnl.lemmatize(sent[i+1][0])),
                'normBigram[0]=%s' % (normalizeWord(sent[i][0]) + " " + normalizeWord(sent[i+1][0])),
                'posBigram[0]=%s' % (sent[i][1] + " " + sent[i+1][1]),
            ])
        if i+2 < len(sent) :
            features.extend([
                'bigram[+1]=%s' % (sent[i+1][0] + " " + sent[i+2][0]),
                'lemBigram[+1]=%s' % (wnl.lemmatize(sent[i+1][0]) + " " + wnl.lemmatize(sent[i+2][0])),
                'normBigram[+1]=%s' % (normalizeWord(sent[i+1][0]) + " " + normalizeWord(sent[i+2][0])),
                'posBigram[+1]=%s' % (sent[i+1][1] + " " + sent[i+2][1]),
            ])

    if len(sent) >= 3: #Then only bigram features exist
        if i > 1 :
            features.extend([
                'trigram[-2]=%s' % (sent[i-2][0] + " " + sent[i-1][0] + " " + sent[i][0]),
            ])
        if i > 0 and i+1 < len(sent):
            features.extend([
                'trigram[-1]=%s' % (sent[i-1][0] + " " + sent[i][0] + " " + sent[i+1][0]),
            ])
        if i+2 < len(sent) :
            features.extend([
                'trigram[0]=%s' % (sent[i][0] + " " + sent[i+1][0] + " " + sent[i+2][0]),
            ])
        if i+3 < len(sent) :
            features.extend([
                'trigram[+1]=%s' % (sent[i+1][0] + " " + sent[i+2][0] + " " + sent[i+3][0]),
            ])

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:postag=' + postag1,
            '+1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('EOS')
        
    return features

def sent2features(sent): #for every sentence extract get feature list
    return [word2features(sent, i) for i in range(len(sent))]
