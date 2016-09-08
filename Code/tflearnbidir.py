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

import tensorflow as tf
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



train_data,file_list = pp.givedata("./data/BioNLP-ST-2016_BB-cat+ner_train","BB-cat(.*)txt")
#dev_data = pp.givedata("./data/BioNLP-ST-2016_BB-cat+ner_dev","BB-cat(.*)txt")


vocab = pp.build_vocab(train_data)

print(len(train_data[1].split()))
trainX = pp.convert_data(train_data,vocab)
# testX = pp.convert_data(dev_data,vocab)


# testX = pad_sequences(testX, maxlen=200, value=0.)
trainX = pad_sequences(trainX, maxlen=200, value=0.)
print ("end here")
trainY = pp.givelabels("./data/BioNLP-ST-2016_BB-cat+ner_train",file_list)
trainY = pad_sequences(trainY, maxlen=200, value=0.)
#print(len(trainX))
#print(len(trainY))
print(len(trainX[5]))
print(len(trainY[5]))
for i in range(0,len(trainX)):
	if len(trainX[i]) == len(trainY[i]):
		continue
	else:
		print(i,len(trainX[i]),len(trainY[i]),file_list[i])
#print(trainX[1])
#print(trainY[1])

# # Converting labels to binary vectors
# #print(trainY)
trainY1=[]
for case in trainY:
	case = to_categorical(case, nb_classes = 5)
	trainY1.append(case)
	# break

# testY1 =[]
# for case in testY:
# 	case = to_categorical(case, nb_classes = 5)
# 	testY1.append(case)

trainY = trainY1
# testY = testY1

def sequence_loss(y_pred, y_true):
    '''
    Loss function for the seq2seq RNN.  Reshape predicted and true (label) tensors, generate dummy weights,
    then use seq2seq.sequence_loss to actually compute the loss function.
    '''
    # if self.verbose > 2: print ("my_sequence_loss y_pred=%s, y_true=%s" % (y_pred, y_true))
    logits = tf.unpack(y_pred, axis=1)		# list of [-1, num_decoder_synbols] elements
    targets = tf.unpack(y_true, axis=1)		# y_true has shape [-1, self.out_seq_len]; unpack to list of self.out_seq_len [-1] elements
    # if self.verbose > 2:
    #     print ("my_sequence_loss logits=%s" % (logits,))
    #     print ("my_sequence_loss targets=%s" % (targets,))
    weights = [tf.ones_like(yp, dtype=tf.float32) for yp in targets]
    # if self.verbose > 4: print ("my_sequence_loss weights=%s" % (weights,))
    sl = seq2seq.sequence_loss(logits, targets, weights)
    # if self.verbose > 2: print ("my_sequence_loss return = %s" % sl)
    return sl

def accuracy(y_pred, y_true, x_in):		# y_pred is [-1, self.out_seq_len, num_decoder_symbols]; y_true is [-1, self.out_seq_len]
    '''
    Compute accuracy of the prediction, based on the true labels.  Use the average number of equal
    values.
    '''
    pred_idx = tf.to_int32(tf.argmax(y_pred, 2))		# [-1, self.out_seq_len]
    # if self.verbose > 2: print ("my_accuracy pred_idx = %s" % pred_idx)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(pred_idx, tf.float32), y_true), tf.float32), name='acc')
    return accuracy

# # Network building
net = input_data(shape=[None, 200])
net = embedding(net, input_dim=20000, output_dim=128)
net = bidirectional_rnn(net, BasicLSTMCell(128), BasicLSTMCell(128), True)
# net = dropout(net, 0.5)
# net = fully_connected(net, 2, activation='softmax')
print("wassup")
net = regression(net, optimizer='adam', loss=sequence_loss, metric=accuracy)
print('hello')
# # Training
model = tflearn.DNN(net, clip_gradients=0., tensorboard_verbose=2)
model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=64)
