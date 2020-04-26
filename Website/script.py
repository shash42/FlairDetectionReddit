import nltk
import ast
import pickle
import re
import praw
import os
import pandas as pd
from nltk.corpus import words
from nltk.corpus import stopwords
from flask import Flask, render_template, request, redirect, url_for, jsonify, json
app = Flask(__name__)

reg_model = pickle.load(open("model.pickle","rb"))

reddit = praw.Reddit(client_id='EGV0l4iNPHX3zw', \
                     client_secret='cfZJqAtGpacEHzjzo6ala9IXhOc', \
                     user_agent='flairdetectorscrape', \
                     username='logisbase2', \
                     password='W3codeforlife')
subreddit = reddit.subreddit('india')

flairs = ['AskIndia', 'Non-Political', 'Scheduled', 'Photography', 'Science/Technology', 'Politics', 'Business/Finance', 'Policy/Economy', 'Sports', 'Food', 'Coronavirus']
label_to_id = {'AskIndia': 0,
 'Non-Political': 1,
 'Scheduled': 2,
 'Photography': 3,
 'Science/Technology': 4,
 'Politics': 5,
 'Business/Finance': 6,
 'Policy/Economy': 7,
 'Sports': 8,
 'Food': 9,
 'Coronavirus': 10}

id_to_label = {v: k for k, v in label_to_id.items()}

def getdata(link):
    global subreddit
    cols = {}
    post = reddit.submission(url=link)
    cols['title'] = post.title
    cols['body'] = post.selftext
    cols["comms_num"] = post.num_comments
    cols["url"] = post.url
    comm_more = None
    if(post.num_comments > 2000):
        comm_more = 3
    elif(post.num_comments > 500):
        comm_more = 7
    elif(post.num_comments > 100):
        comm_more = 15
    char_threshold = 200
    count_threshold = 20
    count = 0
    comment_list = []
    post.comments.replace_more(limit=comm_more)
    comment_queue = post.comments[:]  # Start with top-level comments
    while comment_queue:
        comment = comment_queue.pop(0)
        if(len(comment.body) > char_threshold):
            count+=1
        comment_list.append(comment.body)
        if(count >= count_threshold):
            break
    cols["comments"] = ' '.join(comment_list)
    return cols

stop_words = set(stopwords.words('english'))
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
REPLACE_BY_SPACE_URL = re.compile('[-_]')

def clean_url(row):
    row = REPLACE_BY_SPACE_URL.sub(' ', row)
    initial = (row.split())[0]
    row = ' '.join((row.split())[1:])
    row = (re.split("[,.\-!?:/]+", row))
    initial = initial.split('/')
    initial = initial[len(initial) - 1]
    row = initial + " " + ' '.join(row)
    return row

def clean_text(text):
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = ' '.join(word for word in text.split() if word not in stop_words) # remove stopwords from text
    return text

wordlist = set(words.words())
def only_english(text):
    global wordlist
    rettext = " ".join(w for w in nltk.wordpunct_tokenize(text) if w.lower() in wordlist or not w.isalpha())
    return rettext

def preprocess(data):
    data.fillna("",inplace = True)
    data['title'] = data['title'].apply(clean_text)
    data['body'] = data['body'].apply(clean_text)
    data['comments'] = data['comments'].apply(clean_text)
    data['url'] = data['url'].apply(clean_url)
    return data

def Predict(data):
    feat = data['title'] + data['body'] + data['url'] + data['comments']
    y_pred = reg_model.predict(feat)
    return y_pred

@app.route('/', methods=['POST', 'GET']) 
def root():
    flair = ''
    if request.method == 'POST':
        form = request.form
        link = form['link']
        if(len(link) > 5):    
            temp = {"title":[], "body":[], "comments":[], "url":[], "comms_num": []}
            post = getdata(link)
            for key, value in post.items():
                temp[key].append(value)
            data = pd.DataFrame.from_dict(temp)
            data = preprocess(data)
            y_pred = Predict(data)
            flair = flairs[y_pred[0]]
    return render_template('index.html', flair = flair)

@app.route("/automated_testing", methods=['POST', 'GET'])
def test():
    if request.files:
        file = request.files["upload_file"]
        text = file.read()
        text = str(text.decode('utf-8'))
        text_list = text.split('\n')
        links = [line for line in text_list if line.strip() != ""]
        temp = {"title":[], "body":[], "comments":[], "url":[], "comms_num": []}
        cnt = 0
        print(links)
        for link in links:
            print(cnt)
            cnt+=1
            post = getdata(link)
            for key, value in post.items():
                temp[key].append(value)
        data = pd.DataFrame.from_dict(temp)
        data = preprocess(data)
        y_pred = Predict(data)
        outp = {}
        for i in range(len(links)):
            outp[links[i]] = flairs[y_pred[i]]
        return jsonify(outp)
    else:
        return redirect(url_for('root'))

if __name__ == '__main__':
    app.run(debug=True)



