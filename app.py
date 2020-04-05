#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, jsonify, render_template, request


# In[ ]:


import import_ipynb


# In[ ]:


from Jarvis import ProcessResponse


# In[ ]:


ProcessResponse("hello")


# In[ ]:


app = Flask(__name__)


# In[ ]:


@app.route('/')
def home():
    return render_template('Index_chatBot.html')


# In[ ]:


@app.route('/TalkToBot', methods =['Post'])
def TalkToBot():
    user_res = request.form["user_Response"]
    bot_response = ProcessResponse(user_res)
    return render_template('Index_chatBot.html', prediction_text = bot_response)


# In[ ]:


if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)


# In[ ]:





# In[ ]:




