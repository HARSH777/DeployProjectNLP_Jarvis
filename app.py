#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, jsonify, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:


app = Flask(__name__)


# In[ ]:


@app.route('/')
def home():
    return render_template('Index_chatBot.html')


# In[ ]:


@app.route('/TalkToBot', methods =['Post'])
def TalkToBot():
    user_response = request.form["user_Response"]
    
    
    
    robo_response = ''
    
    finalSentences = pickle.load(open('finalSentence.pkl','rb'))
    
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
        sentences = pickle.load(open('sentences.pkl','rb'))
        robo_response = sentences[indexOfMostSimilarSentence]
        
    finalSentences.remove(user_response)
    
    
    return render_template('Index_chatBot.html', prediction_text = robo_response.strip())


# In[ ]:


if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)


# In[ ]:




