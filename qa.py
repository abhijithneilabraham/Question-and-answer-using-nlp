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
from keras_bert import get_base_dict, get_model, gen_batch_inputs

def tokenize(x):
    return word_tokenize(x) #splitting the words with delimiters#if this doesnt work use python split functions
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
            q=tokenize(q)#used to tokenize the question into keywords such that the related answer can be found from the keyword
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
    flatten=lambda data: reduce(lambda x,y:x+y,data) #x is the word index and y gives the word postion in a matrix and this arrages the answer story in order
    '''lambda is a small anonymous function .
    the reduce function here does the specified function(here lambda) and does apply it in a sequence(here data).
    '''
    data=[(flatten(story),q,answer) for story,q,answer in data if not max_length or len(flatten(story))<maxlength]
    return data

 def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
     xs=[]
     xqs=[]
     ys=[]
     for story,query,answer in data:
         x=[word_idx[w]for w in story]
         xq = [word_idx[w] for w in query]
         #index 0 is reserved
         y=np.zeros(len(word_idx)+1)
         y[word_idx[answer]]=1
         xs.append(x)
         xqs.append(xq)
         ys.append(y)
      return (pad_sequences(xs, maxlen=story_maxlen),
            pad_sequences(xqs, maxlen=query_maxlen), np.array(ys))   #padding means setting all to equal lengths
      
         
RNN=recurrent.LSTM
EMBED_HIDDEN_SIZE = 50
SENT_HIDDEN_SIZE = 100
QUERY_HIDDEN_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 40
print('RNN / Embed / Sent / Query = {}, {}, {}, {}'.format(RNN,
                                                           EMBED_HIDDEN_SIZE,
                                                           SENT_HIDDEN_SIZE,
                                                           QUERY_HIDDEN_SIZE))  
vocab=set()
for story,q,answer in train+test:
    vocab |= set(story + q + [answer])
    
vocab=sorted(vocab)    
vocab_size==len(vocab)+1
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
story_maxlen = max(map(len, (x for x, _, _ in train + test)))
query_maxlen = max(map(len, (x for _, x, _ in train + test)))
x, xq, y = vectorize_stories(train, word_idx, story_maxlen, query_maxlen)
tx, txq, ty = vectorize_stories(test, word_idx, story_maxlen, query_maxlen) 
print('vocab = {}'.format(vocab))
print('x.shape = {}'.format(x.shape))
print('xq.shape = {}'.format(xq.shape))
print('y.shape = {}'.format(y.shape))
print('story_maxlen, query_maxlen = {}, {}'.format(story_maxlen, query_maxlen))

print('Build model...')
sentence = layers.Input(shape=(story_maxlen,), dtype='int32')
encoded_sentence = layers.Embedding(vocab_size, EMBED_HIDDEN_SIZE)(sentence)
encoded_sentence = layers.Dropout(0.3)(encoded_sentence)

question = layers.Input(shape=(query_maxlen,), dtype='int32')
encoded_question = layers.Embedding(vocab_size, EMBED_HIDDEN_SIZE)(question)
encoded_question = layers.Dropout(0.3)(encoded_question)
encoded_question = RNN(EMBED_HIDDEN_SIZE)(encoded_question)
encoded_question = layers.RepeatVector(story_maxlen)(encoded_question)

merged = layers.add([encoded_sentence, encoded_question])
merged = RNN(EMBED_HIDDEN_SIZE)(merged)
merged = layers.Dropout(0.3)(merged)
preds = layers.Dense(vocab_size, activation='softmax')(merged)

model = get_model(
    token_num=len(story_maxlen),
    head_num=5,
    transformer_num=12,
    embed_dim=25,
    feed_forward_dim=100,
    seq_len=20,
    pos_num=20,
    dropout=0.05,
)

def _generator():
    while True:
        yield gen_batch_inputs(
            vocab,
            token_dict,
            token_list,
            seq_len=20,
            mask_rate=0.3,
            swap_sentence_rate=1.0,
        )
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training')
model.fit_generator(
    generator=_generator(),
    steps_per_epoch=1000,
    epochs=100,
    validation_data=_generator(),
    validation_steps=100,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    ],
)

loss, acc = model.evaluate([tx, txq], ty,
                           batch_size=BATCH_SIZE)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))      