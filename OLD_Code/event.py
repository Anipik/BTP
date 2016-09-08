
from collections import defaultdict
import sys
import re
import numpy as np 
from math import log
from sklearn import preprocessing
import os
import nltk
import pickle 
from nltk.tokenize import sent_tokenize, word_tokenize, WordPunctTokenizer
import pycrfsuite
from disease_recog import sent2features
import traceback
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import time
import  gensim,logging
from nltk.corpus import stopwords

ts = time.time()
wnl = WordNetLemmatizer()

stemmer = SnowballStemmer("english")

#model = gensim.models.Word2Vec.load_word2vec_format('./Datasets/GoogleNews-vectors-negative300.bin.gz', binary=True)

class Event:
	def __init__(self):
		self.error1 = 0
		self.error2 = 0

	def getHabitat(self,h):
		return h.split('\t')[-1].lower()

	def getBacteria(self,b):
		return b.split('\t')[-1].lower()

	def w2vFeaturesPositive(self,b,h,doc):
		l = [['lives','inside'],['lives','in','habitat'],['causes','disease','infection'],['isolated','from'],['symbiotic','relationship'],['found'],['present','in','chemical']]
		try:
			sents = sent_tokenize(doc)
			startofb = int(b.split('\t',1)[1].split()[1])
			endofb = (b.split('\t',1)[1].split()[2])
			startofh = int(h.split('\t',1)[1].split()[1])
			endofh = (h.split('\t',1)[1].split()[2])

			if ';' in endofb:
				endofb = endofb.split(';')[0]
			if ';' in endofh:
				endofh = endofh.split(';')[0]
			endofh = int(endofh)
			endofb = int(endofb)

			if startofb > endofh:
				words = WordPunctTokenizer().tokenize(doc[endofh:startofb])
			else:
				words = WordPunctTokenizer().tokenize(doc[endofb:startofh])
			# print words 
			neww = []
			for w in words:
				if len(w) > 1 and w not in stopwords.words('english') and w.isalpha() and w in model.vocab:
					neww.append(w)
			# print neww 
			words = neww
			scores = [0]*len(l)
			if len(words)>0:
				for i,vec in enumerate(l):
					score = model.n_similarity(vec,words)
					scores[i] = score
			return scores
		except Exception,e:
			print str(e)
			traceback.print_exc()
			self.error1+=1
			return [0]*len(l)

	def w2vFeaturesNegative(self,b,h,doc):
		l = [['binds','cell'],['dead']]
		try:
			sents = sent_tokenize(doc)
			startofb = int(b.split('\t',1)[1].split()[1])
			endofb = (b.split('\t',1)[1].split()[2])
			startofh = int(h.split('\t',1)[1].split()[1])
			endofh = (h.split('\t',1)[1].split()[2])

			if ';' in endofb:
				endofb = endofb.split(';')[0]
			if ';' in endofh:
				endofh = endofh.split(';')[0]
			endofh = int(endofh)
			endofb = int(endofb)
			if startofb > endofh:
				words = WordPunctTokenizer().tokenize(doc[endofh:startofb])
			else:
				words = WordPunctTokenizer().tokenize(doc[endofb:startofh])
			# print words 
			neww = []
			for w in words:
				if len(w) > 1 and w not in stopwords.words('english') and w.isalpha() and w in model.vocab:
					neww.append(w)
			# print neww 
			words = neww
			scores = [0]*len(l)
			if len(words)>0:
				for i,vec in enumerate(l):
					score = model.n_similarity(vec,words)
					scores[i] = score
			return scores
		except Exception,e:
			print str(e)
			traceback.print_exc()
			self.error2+=1
			return [0]*len(l)


	def numSentencesInBetween(self,b,h,doc):
		try:
			sents = sent_tokenize(doc)
			startofb = int(b.split('\t',1)[1].split()[1])
			startofh = int(h.split('\t',1)[1].split()[1])
			sentindexofb=-1
			sentindexofh=-1
			cur = 0
			for i,sent in enumerate(sents):
				cur += len(sent)
				if startofb < cur and sentindexofb == -1:
					sentindexofb=i
				if startofh < cur and sentindexofh == -1:
					sentindexofh=i
			return abs(sentindexofb - sentindexofh)
		except Exception,e:
			print str(e)
			traceback.print_exc()
			return 0

	def beforeAfter(self,b,h):
		try:
			startofb = int(b.split('\t',1)[1].split()[1])
			startofh = int(h.split('\t',1)[1].split()[1])
			return 1 if (startofb - startofh) < 0 else 0
		except Exception,e:
			print str(e)
			traceback.print_exc()
			return 0

	def geoOrHabitat(self,h):
		return 1 if h.split('\t',1)[1][0]=='H' else 0

	def numberOfWords(self,b,h,doc):
		try:
			sents = sent_tokenize(doc)
			startofb = int(b.split('\t',1)[1].split()[1])
			startofh = int(h.split('\t',1)[1].split()[1])
			if startofb > startofh:
				return len(word_tokenize(doc[startofh:startofb]))-1
			else:
				return len(word_tokenize(doc[startofb:startofh]))-1
		except Exception,e:
			print str(e)
			traceback.print_exc()
			return 0		

	#For Positive 
	def countKeyWords(self,b,h,doc):
		l = ['isolate','disease','inside','habitat','lives','colonization','commensality','disease','infection','invasion','abscess','found','symbiotic','peptone','galactose','glucose','lactate','acetate']
		trigwords = []
		for w in l:
			trigwords.append(stemmer.stem(w))
			#trigwords.append(wnl.lemmatize(w))
		try:
			sents = sent_tokenize(doc)
			startofb = int(b.split('\t',1)[1].split()[1])
			startofh = int(h.split('\t',1)[1].split()[1])
			if startofb > startofh:
				temp = startofh
				startofh = startofb
				startofb = temp
			#startofb is less
			s = doc[max(0,startofb-50):min(len(doc)-1, startofh+100)]
			words = word_tokenize(s)
			s = []
			result = []
			for w in words:
				s.append(stemmer.stem(w))
				#s.append(wnl.lemmatize(w))
			for t in trigwords:
				if t in s:
					result.append(1)
				else:
					result.append(0)
			#found = set(trigwords).intersection(s)
			return result
		except Exception,e:
			print str(e)
			traceback.print_exc()
			return [0]*len(l)


	def diseaseNamePresent(self,b,h,doc):
		try:
			sents = sent_tokenize(doc)
			startofb = int(b.split('\t',1)[1].split()[1])
			startofh = int(h.split('\t',1)[1].split()[1])
			sentindexofb=-1
			sentindexofh=-1
			cur = 0
			for i,sent in enumerate(sents):
				cur += len(sent)
				if startofb < cur and sentindexofb == -1:
					sentindexofb=i
				if startofh < cur and sentindexofh == -1:
					sentindexofh=i
			if sentindexofb==sentindexofh:	#Belon to same sentence
				tagger = pycrfsuite.Tagger()
				tagger.open('./trainedFiles/ncbidis.crfsuite')
				return int("B-Dis" in tagger.tag(sent2features(nltk.pos_tag(word_tokenize(sents[sentindexofh])))))
			else:
				return 0
		except Exception,e:
			print str(e)
			traceback.print_exc()
			return 0 


	def generatePairs(self,filename):
		bacteria = []
		habgeo = []
		try:
			with open(filename,'r') as f:
				data = f.read().decode("utf8").strip('\n')
				lines = data.split('\n')
				for line in lines:
					items = line.split('\t',1)
					if items[1][0]=='B' or items[1][0]=='H' or items[1][0]=='G':
						if items[1][0] == 'B':
							bacteria.append(line)
						else:
							habgeo.append(line)
		except Exception, e:
			print str(e)
			traceback.print_exc()
		return bacteria,habgeo

	def isRelation(self,b,h,filename):
		#print filename
		with open(filename,'r') as f:
			doc = f.read().decode('utf8').strip('\n')
		if doc == "":
			return 0
		bacteria = b.split('\t')[0]
		habitat = h.split('\t')[0]
		lines = doc.split('\n')
		for line in lines:
			if line[0] != 'R':	#for * case
				continue
			bac = line.split(' ')[1]
			bac = bac.split(':')[1]
			hab = line.split(' ')[-1].split(':')[1]
			if bac == bacteria and hab == habitat:
				return 1
		return 0

	def generateFeatureMatrix(self,mode):
		X = []
		Y = []
		for f in os.listdir(os.getcwd()+"/BioNLP-ST-2016_BB-event_"+mode):
			if f.endswith('.txt'):
				with open("./BioNLP-ST-2016_BB-event_"+mode+'/'+f,'r') as docfile:
					doc = docfile.read().decode('utf8').strip('\n')
				b,h = self.generatePairs("./BioNLP-ST-2016_BB-event_"+mode+"/"+f[:-3]+'a1')
				for bacteria in b:
					for habitat in h:
						feature = []
						#feature.append(self.getBacteria(bacteria))
						#feature.append(self.getHabitat(habitat))
						feature.append(self.numSentencesInBetween(bacteria,habitat,doc))
						feature.append(self.beforeAfter(bacteria,habitat))
						feature.append(self.geoOrHabitat(habitat))
						feature.append(self.numberOfWords(bacteria,habitat,doc))
						feature.extend(self.countKeyWords(bacteria,habitat,doc))
						feature.append(self.diseaseNamePresent(bacteria,habitat,doc))
						feature.extend(self.w2vFeaturesPositive(bacteria,habitat,doc))
						feature.extend(self.w2vFeaturesNegative(bacteria,habitat,doc))
						X.append(feature)
						Y.append(self.isRelation(bacteria,habitat,"./BioNLP-ST-2016_BB-event_"+mode+"/"+f[:-3]+'a2'))
				print sum(Y),f
				#raw_input()
		X = np.asarray(X, dtype=object) #Feature Matrix
	 	Y = np.asarray(Y,dtype=int)
		# for j in range(2):
		# 	le = preprocessing.LabelEncoder()
		# 	le.fit(X[:, j])
		# 	X[:, j] = le.transform(X[:, j])

		for j in range(X.shape[1]):
			X[:, j] = X[:, j].astype(float)
		X = preprocessing.scale(X)
		print X.shape
		pickle.dump(dict(X=X,Y=Y),open("./trainedFiles/feature_"+mode+".pkl",'wb'))

if __name__ == "__main__":
	if len(sys.argv)<2 :
		mode = "train"
	else:
		mode = sys.argv[1]	#test or dev
	e = Event()
	e.generateFeatureMatrix(mode)
	e.generateFeatureMatrix('dev')
	print e.error1,e.error2
	#b,h = e.generatePairs("./BioNLP-ST-2016_BB-event_train/BB-event-1016123.a1")
	#print e.getHabitat(h[0]),e.getBacteria(b[1])
	#print e.numberOfWords(b[1],h[0],"./BioNLP-ST-2016_BB-event_train/BB-event-1016123.txt")
	#print e.countKeyWords(b[0],h[0],"./BioNLP-ST-2016_BB-event_train/BB-event-1016123.txt")
