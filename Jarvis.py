#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# In[8]:


sentences = []
for files in os.listdir('english'):
    chats = open('english/'+files
        , 'r').readlines()
    sentences.append(chats)


# In[9]:


print(sentences[0])


# In[10]:


import re


# In[11]:


paragraph = ""
for sentence in sentences:
    paragraph = paragraph + " ".join(sentence)


# In[12]:


paragraph


# In[13]:


sentences = [re.sub('[^a-zA-Z]', ' ', sentence) for sentence in paragraph.split('\n')]


# In[14]:


sentences


# In[15]:


wordnet = WordNetLemmatizer()


# In[16]:


finalSentences = []
for sentence in sentences:
    sentence = sentence.lower()
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    words = nltk.word_tokenize(sentence)
    words = [wordnet.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    sentence = ' '.join(words)
    finalSentences.append(sentence)


# In[17]:


print(sentences[10])


# In[18]:


def Response(user_response):
    
    user_response = user_response.lower()
    
    robo_response = ''
    
    finalSentences.append(user_response)
    
    tfidf = TfidfVectorizer()
    
    wordTable = tfidf.fit_transform(finalSentences)
    
    # here we have taken -1 because we have append user_response at the last position to get last position we -1....similar to iloc[:,-1] for taking last target variable
    vals = cosine_similarity(wordTable[-1], wordTable)
    
    #val contains score for similarity
    #print(vals)
    
    #Get the index of the most similar text/sentence to the users response
    # here 0 and -2 because of following reason val is list withing list and so 0 indicates index
    # -2 because last index(-1) will have highest similar score which is the user_response itself so we have 
    # taken -2 that is next highest that is second last
    indexOfMostSimilarSentence = vals.argsort()[0][-2]
    
    # above code was to fetch the index of most similar sentence from vals below code is to fetch score of most
    # similar sentence from vals
    flat = vals.flatten()
    
    flat.sort()
    
    score = flat[-2]
    
    if score == 0:
        robo_response = "I apologize, I don't understand."
    else:
        robo_response = sentences[indexOfMostSimilarSentence]
        
    finalSentences.remove(user_response)
        
    return robo_response.strip()


# In[19]:


Response("what do you play")


# In[20]:


finalSentences


# In[21]:


GREETING_INPUTS = ["hi", "hello", "hola", "greetings", "wassup", "hey"]

GREETING_RESPONSES=["howdy", "hi", "hey", "what's good", "hello", "hey there"]


# In[22]:


def Greetings(sentence):
    words = nltk.word_tokenize(sentence)
    for word in words:
        if word.lower() in GREETING_INPUTS:
            return np.random.choice(GREETING_RESPONSES)


# In[23]:


def ProcessResponse(user_response):
    user_response = user_response.lower()
    if not 'bye' in user_response:
        if 'thank' in user_response:
            return "ChatBot: welcome"
        if Greetings(user_response) != None:
            return "ChatBot: "+ Greetings(user_response)
        else: 
            return "ChatBot: "+ Response(user_response)
    else:
        return "ChatBot: see you later!"


# In[24]:


ProcessResponse("hello")


# In[ ]:




