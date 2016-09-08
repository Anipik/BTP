from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
import numpy
import string
from nltk.stem import WordNetLemmatizer

print(sklearn.__version__)
print(numpy.__version__)

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
        'word[-5:]=' + word[-5:],
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

def sent2labels(sent):  #list of labels
    return [label for token, postag, label in sent]

def sent2tokens(sent):  #list of tokens
    return [token for token, postag, label in sent]

def createTrain_sents(mode):
    sents=[]
    with open("tag_"+mode+".out",'r') as output_label:
        text = output_label.read()
        text = text.decode('utf-8')
    tok_list = text.split("\n\n")   #each sentence together
    with open('pos_output_'+mode+'.txt','r') as postag_label:
        postext = postag_label.read()
        postext = postext.decode('utf-8')
    postok_list = postext.split("\n\n")
    i = 0
    while i < len(postok_list):
        words1 = postok_list[i].split("\n")
        words2 = tok_list[i].split("\n")
        sent = []
        for w1,w2 in zip(words1,words2):
            l = w1.split()
            if len(l) != 2:
                continue
            token = l[0]
            pos = l[1]
            if len(w2.split()) <2 :
                pass
                #print w2.split()
            else:
                y_label = w2.split()[1]
            sent.append((token,pos,y_label))
        sents.append(sent)
        i+=1
    return sents

def createOutputFile(file_n,sents,y_pred):
    f = open(file_n,'w')
    #assert()
    for i,sent in enumerate(sents):
        for j,tup in enumerate(sent):
            s =  tup[0]+" "+tup[2]+" "+y_pred[i][j]+'\n'
            s = s.encode('utf-8')
            f.write(s)
        f.write( '\n')
    f.close()

train_sents = createTrain_sents('train')
print "1"
print len(train_sents)
test_sents = createTrain_sents('dev')
print len(test_sents)

X_train = [sent2features(s) for s in train_sents]
print "2"
X_test = [sent2features(s) for s in test_sents]

y_train = [sent2labels(s) for s in train_sents]
print "3"

y_test = [sent2labels(s) for s in test_sents]
print "4"

#Train the model
trainer = pycrfsuite.Trainer(verbose=False)

for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)
print "5"
trainer.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 50,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})

print trainer.params()

trainer.train('./trainedFiles/conll2002-esp.crfsuite')

#make predictions
tagger = pycrfsuite.Tagger()
tagger.open('./trainedFiles/conll2002-esp.crfsuite')


f1 = open('./first100.txt', 'w')
i = 0
while i<100:
    example_sent = test_sents[i]
    f1.write(( ' '.join(sent2tokens(example_sent))).encode('utf-8'))
    f1.write('\n')
    f1.write("Predicted: ")
    f1.write(( ' '.join(tagger.tag(sent2features(example_sent)))).encode('utf-8'))
    f1.write('\n')
    f1.write("Correct:  ")
    f1.write(( ' '.join(sent2labels(example_sent))).encode('utf-8'))
    f1.write('\n\n')
    i+=1

#Evaluate the model

def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.

    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()

    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])


    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )
print "6"
y_pred = [tagger.tag(xseq) for xseq in X_test]

print(bio_classification_report(y_test, y_pred))

from collections import Counter
info = tagger.info()
createOutputFile("neroutput.txt", test_sents,y_pred)

#In [19]:



#http://nbviewer.jupyter.org/github/tpeng/python-crfsuite/blob/master/examples/CoNLL%202002.ipynb
