#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer


# In[2]:


train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")


# In[3]:


nltk.download('state_union')


# In[4]:


custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)


# In[5]:


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


# In[ ]:


def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            chunked.draw()     

    except Exception as e:
        print(str(e))

process_content()


# In[ ]:





# In[ ]:




