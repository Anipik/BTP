import sys

formated_directory = './formated/'

def formatw(w):
	if w!="O":
		return w.split('_')[1]+'-'+w.split('_')[0]
	return w
	
def convert(filen):
	with open(filen,'r') as f, open(formated_directory+"f_"+filen,'w') as w:
		lines = f.read().strip('\n').split('\n')
		for line in lines:
			newl = ""
			words = line.split()
			assert(len(words)==3 or len(words)==0 ),"split length should be 0 or 3"
			if len(words)==3:
				newl = words[0]+' '+formatw(words[1])+' '+formatw(words[2])+'\n'
				w.write(newl)
			else:
				w.write('\n')

def exact_Matching(filen):
	with open(filen,'r') as f:
		lines = f.read().strip('\n').split('\n')
		for line in lines:
			words = line.split()
			

if __name__ == '__main__':
    print sys.argv[1]
    convert(sys.argv[1])