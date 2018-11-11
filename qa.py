from __future__ import print_function
from functools import reduce
import re
import tarfile

import numpy as np

from keras.utils.data_utils import get_file
from keras.layers.emdeddings import Embedding
from keras import layers
from keras.layers import recurrent
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

tokenizer=Tokenizer()

def parse_stories(lines,only_supporting=False):
    data=[]
    story=[]
    for line in lines:
        line=line.decode('utf-8').strip()
        nid,line=line.split(' ',1)
        nid=int(nid)
        