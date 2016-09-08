
from collections import defaultdict
import sys
import re
from math import log
from nltk import bigrams
from nltk import trigrams
import os
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from difflib import SequenceMatcher

wnl = WordNetLemmatizer()
stemmer = SnowballStemmer("english")

class Cat:
    def __init__(self):
        self.path_to_train_folder = "./BioNLP-ST-2016_BB-cat_train/"
        self.onto = {}
        self.taxo = {}
        self.inv_onto = defaultdict(list)
        self.inv_taxo = defaultdict(list)
        self.dic_bac = defaultdict(list)    #Dictionary for bac taxo mapping Training set
        self.dic_hab = defaultdict(list)    #Dictionary for hab taxo mapping Training set
        self.preprocessTrainData()

    def preprocessTrainData(self):
        for f in os.listdir(self.path_to_train_folder):
            if f.endswith('a1'):
                with open(self.path_to_train_folder+f,'r') as f1,open(self.path_to_train_folder+f[:-1]+'2','r') as f2:
                    data1 = f1.read().decode("utf8").strip('\n')
                    data2 = f2.read().decode("utf8").strip('\n')
                    lines1 = data1.split('\n')  #a1 file
                    lines2 = data2.split('\n')  #a2 files
                    if data2=="":   #No Categories present
                        continue
                    tdict = dict()
                    for line in lines1:
                        items = line.split('\t')
                        if items[1][0] == 'H' or items[1][0] == 'B':
                            tdict[items[0]] = items[2]  
                    for line in lines2:
                        entity = line.split('\t')[1]
                        key = tdict[entity.split(' ')[1].split(':')[1]].lower()
                        value = entity.split(' ')[-1].split(':',1)[1]
                        if entity[0] == 'O' :
                            self.dic_hab[key].append(value) #habitat to OBT:000001
                        else:
                            self.dic_bac[key].append(value) #bac to 23423


    def getLemmatizedString(self,s):
        words = s.split()
        newwords = []
        for w in words:
            newwords.append(wnl.lemmatize(w))
        news = " ".join(newwords)
        return news   

    def getStemmedString(self,s):     
        words = s.split()
        newwords = []
        for w in words:
            if w not in stopwords.words('english'): #Remove Stop Words
                newwords.append(stemmer.stem(w))
        if not newwords:    #If newwords is empty list return same string
            return s
        news = " ".join(newwords)
        return news

    def lemmatizedExpansion(self,s,val):
        news = self.getLemmatizedString(s)
        if s != news:
            #print s," ---- ",news
            self.onto[news] = val
            self.inv_onto[val].append(news)

    def stemmedExpansion(self,s,val):
        news = self.getStemmedString(s)
        if s != news:
            #print s," ---- ",news
            self.onto[news] = val
            self.inv_onto[val].append(news)

    def makeOnto(self):
        with open('./OntoBiotope_BioNLP-ST-2016.obo','r') as f:
            data = f.read().decode("utf8") #newly added .decode("utf8")
            l = data.split("[Term]")[1:]
            for o in l:
                o =o.strip('\n')
                lines = o.split("\n")
                val = lines[0].split()[1]
                name = (lines[1].split(None,1)[1]).lower()
                self.onto[name] = val
                self.inv_onto[val].append(name)
                #Expand Dictionary
                self.stemmedExpansion(name,val)

                for line in lines[2:]:
                    items = line.split()
                    if items[0][0]=='n' or items[0][0]=='s': #name or synonym
                        k = line.split('\"')[1].lower()
                        self.onto[k] = val
                        self.inv_onto[val].append(k)
                        #Expand Dictionary
                        self.stemmedExpansion(k,val)
                    else:
                        break

    def makeTaxo(self):
        with open('./taxdmp/names.dmp','r') as f:
            data = f.read().decode("utf8")
            lines = data.split('\n')[:-1]
            for line in lines:
                items = line.split('\t|\t')
                bacteria = items[1].lower()
                self.taxo[bacteria]=items[0]
                self.inv_taxo[items[0]].append(bacteria)

    def get(self):
        while True:
            x = raw_input("A")
            print self.taxo[x]


    def getMatch(self,filename, predlist,actual):
        print actual
        entity = actual.split(' ')[1].split(':')[1]

        y_true = self.inv_onto[actual.split(' ')[-1].split(':',1)[1]]
        #print entity,y_true,actual.split(' ')[-1].split(':',1)[1]
        for pred in predlist:
            if pred[0]=='O' and entity == pred.split(' ')[1].split(':')[1]:
                y_pred = self.inv_onto[pred.split(' ')[-1].split(':',1)[1]]
                break
        print "actual: ",y_true
        print "pred: ",y_pred
        print entity,filename

    def Evaluation(self, mode):
        directory = "./BB_sol_cat_"+mode
        match_onto = 0
        notmatch_onto = 0
        match_taxo = 0
        notmatch_taxo = 0
        for f in os.listdir(os.getcwd()+"/BioNLP-ST-2016_BB-cat_"+mode):
            if f.endswith('.a1'):   #Open a2 file and our predicted .sol file
                with open("./BioNLP-ST-2016_BB-cat_"+mode+"/"+f[:-1]+'2','r') as f1, open(directory+"/"+f+".sol",'r') as f2:
                    data1 = f1.read().decode("utf8").strip('\n')
                    data2 = f2.read().decode("utf8").strip('\n')
                    lines1 = data1.split('\n')  #True
                    lines2 = data2.split('\n')  #Predicted
                    if data1=="":
                        continue
                    lines1 = [ line.split('\t')[1] for line in lines1]
                    lines2 = [ line.split('\t')[1] for line in lines2]

                    for line in lines1:
                        isOnto = False
                        if line[0] == 'O':
                            isOnto = True
                        if line in lines2: #Find the True Category in the Predicted Category list
                            if isOnto:
                                match_onto+=1 
                                #self.getMatch(f,lines2,line)
                            else:
                                match_taxo+=1
                        else:
                            if isOnto:
                                notmatch_onto+=1
                                #self.getMatch(f,lines2,line) 
                            else:
                                notmatch_taxo+=1 

        print "\nFound Taxo: ",match_taxo
        print "NotFound Taxo: ",notmatch_taxo
        print "Found Onto: ", match_onto
        print "NotFound Onto: ",notmatch_onto
        print "Accuracy Onto: ", float(match_onto)/float(match_onto+notmatch_onto)
        print "Accuracy Taxo: ", float(match_taxo)/float(match_taxo+notmatch_taxo)



    def mapEntity(self,mode):
        directory = "./BB_sol_cat_"+mode
        if not os.path.exists(directory):
            os.makedirs(directory)
        found_onto = 0
        notfound_onto = 0
        found_taxo = 0
        notfound_taxo = 0
        for f in os.listdir(os.getcwd()+"/BioNLP-ST-2016_BB-cat_"+mode):
            if f.endswith('.a1'):
                with open("./BioNLP-ST-2016_BB-cat_"+mode+"/"+f,'r') as corpus_file,open(directory+"/"+f+".sol",'w')as o_file:#,open(directory+"/"+f[:-1]):
                    data = corpus_file.read().decode("utf8").strip('\n')   #Remove unnecessary endlines in the file at begin and end
                    lines = data.split("\n") #Skip first two (title and abstract)
                    N=0
                    for line in lines:
                        items = line.split('\t')
                        entity = items[-1].lower()
                        if items[1][0] == 'B':    #If Tx is Bacteria or Habitat and not Title and Paragraph
                            N+=1
                            if entity in self.taxo.keys():
                                found_taxo+=1
                                o_file.write("N"+str(N)+'\t'+'NCBI_Taxonomy Annotation:'+items[0]+' Referent:'+self.taxo[entity]+'\n')
                            else:
                                maxv = 0.0
                                sim = ""
                                for k in self.dic_bac.keys():
                                    v = SequenceMatcher(None, entity, k).ratio()
                                    if v > maxv:
                                        maxv = v
                                        sim = k 
                                found_taxo+=1
                                obs = self.dic_bac[sim]  # is a list
                                for ob in obs:
                                    o_file.write("N"+str(N)+'\t'+'NCBI_Taxonomy Annotation:'+items[0]+' Referent:'+ob+'\n')
                                """
                                else:
                                    notfound_taxo+=1
                                    o_file.write("N"+str(N)+'\t'+'NCBI_Taxonomy Annotation:'+items[0]+' Referent:'+'NotFound'+'\n')
                                """
                        #Habitat
                        elif items[1][0] =='H':
                            N+=1
                            if entity in self.onto.keys():
                                found_onto+=1
                                o_file.write("N"+str(N)+'\t'+'OntoBiotope Annotation:'+items[0]+' Referent:'+self.onto[entity]+'\n')
                            elif self.getStemmedString(entity) in self.onto.keys():
                                found_onto+=1
                                #print entity, self.getStemmedString(entity)
                                o_file.write("N"+str(N)+'\t'+'OntoBiotope Annotation:'+items[0]+' Referent:'+self.onto[self.getStemmedString(entity)]+'\n')    #Two Times                           
                            else:
                                #Similarity with habitats in train data
                                #entity = self.getStemmedString(entity)
                                maxv = 0.0
                                sim = ""
                                for k in self.dic_hab.keys():
                                    v = SequenceMatcher(None, entity, k).ratio()
                                    if v > maxv:
                                        maxv = v
                                        sim = k 
                                #print entity," - ---------- ",sim
                                entity = self.getStemmedString(entity)
                                simk = sim
                                for k in self.onto.keys():
                                    v = SequenceMatcher(None, entity, k).ratio()
                                    if v > maxv:
                                        maxv = v
                                        simk = k                               


                                if simk==sim:   #higher similarity from training data
                                    found_onto+=1
                                    obs = self.dic_hab[sim]  # is a list
                                    for ob in obs:
                                        o_file.write("N"+str(N)+'\t'+'OntoBiotope Annotation:'+items[0]+' Referent:'+ob+'\n')
                                    #print entity, "------", sim
                                    #raw_input()
                                else:
                                    found_onto+=1
                                    o_file.write("N"+str(N)+'\t'+'OntoBiotope Annotation:'+items[0]+' Referent:'+self.onto[simk]+'\n')
                                    
                            

                                """
                                #Similarity with keys in onto file
                                entity = self.getStemmedString(entity)
                                maxv = 0.0
                                sim = ""
                                for k in self.onto.keys():
                                    v = SequenceMatcher(None, entity, k).ratio()
                                    if v > maxv:
                                        maxv = v
                                        sim = k 
                                #print entity," - ---------- ",sim
                                if maxv > 0.0:
                                    found_onto+=1
                                    o_file.write("N"+str(N)+'\t'+'OntoBiotope Annotation:'+items[0]+' Referent:'+self.onto[sim]+'\n')
                                    #print entity, "------", sim
                                    #raw_input()
                                else:
                                    notfound_onto+=1
                                    o_file.write("N"+str(N)+'\t'+'OntoBiotope Annotation:'+items[0]+' Referent:'+'NotFound'+'\n')
                                """


        print "Found Taxo: ",found_taxo
        print "NotFound Taxo: ",notfound_taxo
        print "Found Onto: ", found_onto
        print "NotFound Onto: ",notfound_onto

if __name__ == "__main__":
    #train_filename = sys.argv[1]
    #test_filename = sys.argv[2]
    mode = sys.argv[1]
    c = Cat()
    c.makeOnto()
    print "OntoBiotope done"
    c.makeTaxo()
    print "NCBIDone"
    c.mapEntity(mode)
    c.Evaluation(mode)
#Make Dictionary for OntoBiotope
#Make Dictionary for bac-names

