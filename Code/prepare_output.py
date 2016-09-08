import tensorflow as tf
import os, sys
reload(sys)
sys.setdefaultencoding('utf-8')
tags = ['HAB', 'HAB_C', 'BAC', 'BAC_C', 'GEO', 'GEO_C']
tampered = []
def convert_output(path, filenames):
    outputs = []
    for filename in filenames:
        # filename = 'BB-cat+ner-6107735'
        # print(filename)
        with tf.gfile.GFile(path+'/'+filename+'.txt', "r") as f:
            with tf.gfile.GFile(path+'/'+filename+'.a2', "r") as f1:
              text = f.read().decode('utf-8')
            #   print (text.split())
              init = len(text.split())
              a2 = f1.read().splitlines()
              red = 0
              end = 0
              for line in a2:
                  if line[0] == 'T':
                      split_line = line.split()
                      if ';' in split_line[3]:
                          split_line.remove(split_line[3]);
                          tampered.append(filename)
                      left = int(split_line[2]) - red
                      right = int(split_line[3]) - red
                      if int(split_line[3]) <= end:
                          continue
                      end = int(split_line[3])
                    #   print (text[left:right] + " : " + split_line[1][0:3].upper() + " " + str(red))
                    #   print (text)
                      red = red + len(text)
                      if text[left-1] not in [' ', '\n']:
                          left -= 1;
                      if text[right] not in [' ', '\n']:
                          right += 1;
                      text = text[:left] + split_line[1][0:3].upper() + ''.join([' ' + split_line[1][0:3].upper() + '_C']*len(split_line[5:])) + text[right:]
                      red = red - len(text)
              new_text = ['OTH' if word not in tags else word for word in text.strip().split()]
        if init != len(text.split()):
            print (filename)
        # print(text.split())
        # print(' '.join(new_text))
        # with tf.gfile.GFile(path+'/'+filename+'.out', "w") as f:
        #     f.write((' '.join(new_text)).encode('utf-8'))
        # break
    # return outputs

path = 'data/BioNLP-ST-2016_BB-cat+ner_train'
filenames = []
for file in os.listdir(path):
    if file.endswith(".txt"):
        filenames.append(file[:-4])
# print(convert_output(path,filenames))
convert_output(path, filenames)
# print len(out)
