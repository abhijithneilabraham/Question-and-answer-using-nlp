from __future__ import print_function
from functools import reduce
import re
import tarfile

import numpy as np

from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras import layers
from keras.layers import recurrent
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
def tokenize(x):
    return word_tokenize(x) #splitting the words with delimiters
def parse_stories(lines,only_supporting=False):#for keeping only the supporting answer
    data=[]
    story=[]
    for line in lines:
        line=line.decode('utf-8').strip()#this makes utf-8 decoding into some binary values for the incoming text 
        nid,line=line.split(' ',1)#the split function has the second parameter maxsplit defined,which makes the total array split into two parts only
        nid=int(nid)  #so ,nid and line are taken as two variables 
        if nid==1:
            story= []
        if '\t' in line:
            q,a,supporting=line.split('\t')
            q=tokenize(q)
            if only_supporting: #selecting the related substory only
                supporting=map(int,supporting.split()) #map applies a function to each item in the iterable
                substory=[story[i-1] for i in supporting]
            else:
                [x for x in story if x] #x substory is derived from story
            data.append((substory,q,a))
            story.append('')
        else:
            sent=tokenize(line)
            story.append(sent)
            
    return data      
def get_stories(f, only_supporting=False, max_length=None):
     '''Given a file name, read the file, retrieve the stories,
    and then convert the sentences into a single story.

    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    '''
    data=parse_stories(f.readlines(),only_supporting=only_supporting)
    flatten=lambda data: reduce(lambda x,y:x+y,data) 
    '''lambda is a small anonymous function 
    the reduce function here does the specified function(here lambda) and does apply it in a sequence.
    '''
    
    