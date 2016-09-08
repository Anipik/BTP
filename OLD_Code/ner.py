import os
import re
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

rest = ""

def tagOverlap(entity,name):
	global rest
	t = tokenizeSent(name,0)[0]
	t.split("\n")[:-1]
	for i,n in enumerate(t):
		if i == 0:
			rest+=((n.split('\t')[0]+" "+ entity +"_B\n").encode('utf-8'))
		else:
			rest+=((n.split('\t')[0]+" "+ entity +"_I\n").encode('utf-8'))
	rest+="\n"

def tokenizeSent(sent,offset):
	prev = 0
	tokens = ""
	for i,s in enumerate(sent):
		if not s.isalnum():
			if prev < i:
				tokens += sent[prev:i] + "\t" + str(prev + offset) + "\t" + str(i-1 + offset)+"\n"
			prev = i+1
			if s != ' ' and s != '\n':
				tokens += s + "\t" + str(i + offset) + "\t" + str(i + offset)+"\n"
	if prev != len(sent):
		tokens += sent[prev:i] +  "\t" + str(prev + offset) + "\t" + str(len(sent)-1 + offset)+"\n"
	return tokens,offset+len(sent)

def posTag(mode):
	f = open('pos_output_'+mode+'.txt', 'w')
	with open("tag_"+mode+".out","r") as tag_file:
		sents = tag_file.read().decode('utf-8').split("\n\n")
		for sent in sents:
			words = []
			for s in sent.split("\n"):
				if len(s.split())>0:
					words.append(s.split()[0])
			#print words
			x = nltk.pos_tag(words)
			for tokens in x:
				f.write(tokens[0].encode('utf-8')+" "+tokens[1].encode('utf-8')+'\n')
			f.write("\n")
	f.close()

def labelTags(mode):	#train or dev
	print os.getcwd()
	tagfile = open("tag_"+mode+".out","w")
	for f in os.listdir(os.getcwd()+"/BioNLP-ST-2016_BB-cat+ner_"+mode):
		if f.endswith('.txt'):
			print "Processing file "+f
			with open("./BioNLP-ST-2016_BB-cat+ner_"+mode+"/"+f,'r') as corpus_file:
				lines = corpus_file.read()
				lines = lines.decode('utf-8').split("\n")
				sents=[]
				for line in lines:
					if len(line) > 0:
						sents.extend(sent_tokenize(line))
				outfile = open('./tokenized_'+mode+"/"+f+'.tok','w')
				offset=-1
				for i,sent in enumerate(sents):
					l,offset = tokenizeSent(sent,offset+1)
					l = l.encode('utf-8')
					if i==len(sents)-1:
						outfile.write(l)
					else:
						outfile.write(l+"\n")
				outfile.close()
			with open("./BioNLP-ST-2016_BB-cat+ner_"+mode+"/"+f[:-3]+"a2",'r') as tagged_file, open('./tokenized_'+mode+"/"+f+'.tok',"r") as tok_file:
				lines_r = tagged_file.read().decode('utf-8').split("\n")
				lines_tok = tok_file.read().decode('utf-8').split("\n")
				tok_index = 0
				for line in lines_r:
					if line == "":
						continue
					if line[0] != 'T':
						while tok_index < len(lines_tok):
							if lines_tok[tok_index] == "": #for only \n
								tagfile.write(("\n").encode('utf-8'))
							else:
								tagfile.write((lines_tok[tok_index].split('\t')[0]+" O\n").encode('utf-8'))
							tok_index += 1
						break
					else :
						l = line.split("\t")
						name = l[-1]
						entity = l[1].split()[0]
						begin_i = l[1].split()[1]
						end_i = l[1].split()[2]
						if ';' in begin_i or ';' in end_i:
							#tagOverlap(entity,name)
							#tok_index += 1
							continue

						begin_i = int(begin_i)
						end_i = int(end_i)

						while tok_index < len(lines_tok) and lines_tok[tok_index] == "":
							tagfile.write(("\n").encode('utf-8'))
							tok_index += 1
						while (tok_index < len(lines_tok)) and int(lines_tok[tok_index].split('\t')[1]) < begin_i:
							tagfile.write((lines_tok[tok_index].split('\t')[0]+" O\n").encode('utf-8'))
							tok_index += 1
							while tok_index < len(lines_tok) and lines_tok[tok_index] == "":
								tagfile.write(("\n").encode('utf-8'))
								tok_index += 1

						if tok_index >= len(lines_tok):
							break
						#Match found, issue
						if begin_i == int(lines_tok[tok_index].split('\t')[1]):
							name_split = tokenizeSent(name,0)[0].split("\n")[:-1]
							#print name_split
							for i,n in enumerate(name_split):
								if i == 0:
									tagfile.write((lines_tok[tok_index].split('\t')[0]+" "+ entity +"_B\n").encode('utf-8'))
								else:
									tagfile.write((lines_tok[tok_index].split('\t')[0]+" "+ entity +"_I\n").encode('utf-8'))
								tok_index += 1
						else:
							print f,line
			#raw_input("nextfile")
	# tagfile.write(rest)
	tagfile.close()






labelTags('train')
labelTags('dev')
posTag('train')
posTag('dev')
