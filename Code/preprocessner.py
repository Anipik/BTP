import os
import re


def givedata(path,exp):
	totaldata = []
	file_list = []
	for filename in os.listdir(path):
		if re.match(exp, filename):
			#print(filename)
			file_list.append(filename[:-4])
			data=""
			with open(os.path.join(path, filename), 'r') as f:
				#print(filename)
			 	
			 	for line in f:
			 		data = data + line
			totaldata.append(data)
	return totaldata,file_list


def givelabels(path,file_list):
	totaldata = []
	output = { 'OTH':0, 'BAC':1,'BAC_C':2,'HAB':3,'HAB_C':4}
	for filename in file_list:
		#print(filename)
		with open(os.path.join(path, filename+'.out'), 'r') as f:
			for line in f:
				line = line.strip().split()
			 	k=[]
			 	for word in line:
			 		k.append(output[word])
		totaldata.append(k)
	return totaldata


def build_vocab(word_list_list):
	vocab=[]
	for word_list in word_list_list:
		word_list.strip()
		words = word_list.split()
		vocab = vocab+words
		vocab = list(set(vocab))
	return vocab

def convert_data(data_list,vocab):
	word_to_num = { ch:i for i,ch in enumerate(vocab) }
	num_to_word = { i:ch for i,ch in enumerate(vocab) }

	result=[]
	for data in data_list:
		data= data.split()
		dev_set=[]
		for word in data:
			if word in word_to_num:
				dev_set.append(word_to_num[word])
			else:
				dev_set.append(len(vocab))
		result.append(dev_set)


	return result
