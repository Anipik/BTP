Dependencies:
Sklearn version 0.17.1
Python version 2.7.6
-----------------------

-----------------------
Folders to be downloaded:
1.
Download this folder inside Code folder
https://drive.google.com/file/d/0B5Rh14nCn1JjbTRrX09IZ2w4dlk/view
It contains names.dmp file used for NCBI taxonomy

2.
Download Word2Vec from
https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
keep it in ./Datasets folder only
-------------------------------------------------------------------
Short Running Commands:
$ python dep.py dev			(entity detection results)
$ python cat.py dev			(categorization results)
$ python livesInClassification.py (event detection results)


Files and execution description:
-----------------------------------------------
Entity Recognition

1 ner.py 
performs tokenization and preprocessing and pos tagging. It takes time to generate pos tagging. So you may not wish to run it. 
The input of this file is:
BioNLP-ST-2016_BB-cat+ner_dev folder that contains document abstracts
BioNLP-ST-2016_BB-cat+ner_train folder that contains document abstracts

The output of this file is:
tag_dev.out: tuple <token, tag>  used for evaluating CRF
tag_train.out: tuple <token, tag> used for training CRF
pos_output_dev.txt: POS tagging of tokens in dev data
pos_output_train.txt: POS tagging of tokens in train data



2. dep.py
run by:
$ python dep.py dev    
dev is parameter sys.argv[1]
Trains the CRF on test data and generates output on dev data
The input of this file is:
tag_dev.out, tag_train.out, pos_output_dev.txt, pos_output_train.txt used for training CRF

The output of this file is:
./trainedFiles/conll2002-esp.crfsuite: The trained file of CRF
neroutput.txt: The output of tagging on documents in BioNLP-ST-2016_BB-cat+ner_dev

3. changeFormat.py
run by:
$python changeFormat.py neroutput.txt
To view exact boundary Matching results. makes compatibility changes i.e. converting Bacteria_B to B-Bacteria

Output: f_neroutput.txt in ./formated/ folder

Now go inside formated folder and run the command to view precision, recall, accuracy:
$ perl connlleval.pl < f_neroutput.txt


--------------------------------
Entity Categorization
1. cat.py
run by
$ python cat.py dev
Input Files:
./taxdmp/name.dmp : Folder taxdmp should be created after downloading 1. and extracting in Codes folder
./OntoBiotope_BioNLP-ST-2016.obo : Ontology file
BioNLP-ST-2016_BB-cat_dev folder that contains document abstracts
BioNLP-ST-2016_BB-cat_train folder that contains document abstracts

Output:
./BB_sol_cat_dev : Contains our output for categorization


------------------------------------
Event Detection

1. event.py to build the feature matrix for the classification task. (If you don't want to do that skip to 2.)
run by
$ python event.py
Input Files:
BioNLP-ST-2016_BB-event_dev folder that contains document abstracts
BioNLP-ST-2016_BB-event_train folder that contains document abstracts

Output Files:
./trainedFiles/feature_dev.pkl 
./trainedFiles/feature_train.pkl

2. livesInClassification.py (evaluation of RandomForest, KNN, SVM)
run by 
$ python livesInClassification.py
Input Files:
Output Files:
./trainedFiles/feature_dev.pkl 
./trainedFiles/feature_train.pkl

Output on terminal Classification Report

-------------
3. Clustering results can be found by running
$ python livesinClassification_ani.py
It outputs the evaluation scores for the two clusters - one after the other. The scores are for each classifier.
-------------
Other Files
disease_recog.py : used for disease recognition used as a feature in event.py
./trainedFiles/ncbidis.crfsuite : trained file for disese recognition

