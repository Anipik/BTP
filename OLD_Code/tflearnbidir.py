"""
Simple example using LSTM recurrent neural network to classify IMDB
sentiment dataset.
References:
    - Long Short Term Memory, Sepp Hochreiter & Jurgen Schmidhuber, Neural
    Computation 9(8): 1735-1780, 1997.
    - Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng,
    and Christopher Potts. (2011). Learning Word Vectors for Sentiment
    Analysis. The 49th Annual Meeting of the Association for Computational
    Linguistics (ACL 2011).
Links:
    - http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
    - http://ai.stanford.edu/~amaas/data/sentiment/
"""

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.embedding_ops import embedding
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell
from tflearn.layers.estimator import regression

import preprocessner as pp
from random import randint
# IMDB Dataset loading
# train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000,
#                                 valid_portion=0.1)
# trainX, trainY = train
# testX, testY = test



train_data = pp.givedata("/home/sun/Dropbox/Aneesh-Aniruddh/Code/BioNLP-ST-2016_BB-cat+ner_train","BB-cat(.*)txt")
dev_data = pp.givedata("/home/sun/Dropbox/Aneesh-Aniruddh/Code/BioNLP-ST-2016_BB-cat+ner_dev","BB-cat(.*)txt")


vocab = pp.build_vocab(train_data)

#print(len(train_data[1].split()))
trainX = pp.convert_data(train_data,vocab)
testX = pp.convert_data(dev_data,vocab)


testX = pad_sequences(testX, maxlen=200, value=0.)

trainY = pp.givelabels("/home/sun/Dropbox/Aneesh-Aniruddh/Code/BioNLP-ST-2016_BB-cat+ner_train","BB-cat(.*)out")
#print(len(trainX))
#print(len(trainY))
print(len(trainX[1]))
print(len(trainY[1]))
#print(trainX[1])
#print(trainY[1])

# # Converting labels to binary vectors
# #print(trainY)
# trainY1=[]
# for case in trainY:
# 	case = to_categorical(case, nb_classes = 5)
# 	trainY1.append(case)
# 	break

# testY1 =[]
# for case in testY:
# 	case = to_categorical(case, nb_classes = 5)
# 	testY1.append(case)

# #trainY = trainY1
#testY = testY1 

# # # Network building
# net = input_data(shape=[None, 200])
# net = embedding(net, input_dim=20000, output_dim=128)
# net = bidirectional_rnn(net, BasicLSTMCell(128), BasicLSTMCell(128))
# net = dropout(net, 0.5)
# net = fully_connected(net, 1000, activation='softmax')
# net = regression(net, optimizer='adam', loss='categorical_crossentropy')

# # # Training
# model = tflearn.DNN(net, clip_gradients=0., tensorboard_verbose=2)
# model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=64)