import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.request import urlretrieve
from PyPDF2 import PdfReader
#from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import seaborn as sns
import random

links = []
outfiles = []

def scrape():    
    response = requests.get('https://proceedings.mlr.press/v202/', stream=True)
    soup = BeautifulSoup(response.text, features="lxml")

    for link in soup.find_all('a'): 
        if ".pdf" in str(link): 
            links.append(link.get('href'))
    print(links)
    print(len(links))
    i = 1
    
  
    for link in links:
        outfile = os.path.split(link)[1]
        outfiles.append(outfile)
        with open(outfile, 'w') as fp:
            urlretrieve(link, outfile)
    del response
    #print('still scraping')

scrape()

allwords = []
i = 0
for file in outfiles:
    i+=1
    print(i)
    try:
        reader = PdfReader(file)
        #print(len(reader.pages))
        page = reader.pages[0]
        text = page.extract_text()
        # only letters and spaces
        text = re.sub("\d+", " ", text)
        text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
        text = text.lower()
        text = str.replace(text, '\n', ' ')
        #note above preprocessing was done before discovering countvectorizer so some is redundant
        #print(text)
        allwords.append(text)
    except:
        print(file)

with open("allwords.txt", 'w') as fp:
    for i in range(len(allwords)):
        fp.write(allwords[i])

#Part1

vectorizer = CountVectorizer(lowercase=True)
vectorizer.fit(allwords)
print("d fit")
cv = vectorizer.fit_transform(allwords)
print("done cv")
#print(cv)
#print(vectorizer.get_feature_names_out())
df = pd.DataFrame(cv.toarray(), columns=vectorizer.get_feature_names())
s = df.sum(axis=0)
top_words = s.sort_values(ascending=False)
print(top_words[0:10])
#above gives no useful info so...
#test_ a later addition with discorvery of stop_words
test_vectorizer = CountVectorizer(lowercase=True, stop_words='english')
test_vectorizer.fit(allwords)
test_cv = test_vectorizer.fit_transform(allwords)
test_df = pd.DataFrame(test_cv.toarray(), columns=test_vectorizer.get_feature_names())
t_s = test_df.sum(axis=0)
top_words_wstop = t_s.sort_values(ascending=False)

print(top_words_wstop[0:10])


#Part 2
#P(W=w|P=i)
#remeber to add axis= 0 so per paper
p = df.divide(df.sum(axis=1),axis=0)
#P(W=w)
pw =p.sum(axis=0)*(1.0/df.shape[0]) 
H = -pw.multiply(np.log(pw)).sum()
print(H)

p = test_df.divide(df.sum(axis=1),axis=0)
pw =p.sum(axis=0)*(1.0/test_df.shape[0]) 
H = -pw.multiply(np.log(pw)).sum()
print(H)

#Part 3
#length of paragraph chosen randomly
paragraph = ''
for i in range(random.randint(100,200)):
    word = np.random.multinomial(1,pw,1).argmax()
    if word != 103853: #weird bug fix
        #print(pw.index.values[word])
        #print(word)
        paragraph = paragraph+' '+ pw.index.values[word]
    else: i+=1
print(paragraph)