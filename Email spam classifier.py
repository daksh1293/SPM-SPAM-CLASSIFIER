#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('Spam.csv')


# In[3]:


df.sample(5)


# In[4]:


df.shape


# ## DATA CLEANING
# 

# In[5]:


df.info()


# In[6]:


## drop last 3 columns
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace=True)


# In[7]:


df.info()


# In[8]:


#renaming the cols
df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
df.sample(5)


# In[9]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[10]:


df['target'] = encoder.fit_transform(df['target'])


# In[11]:


df.head()


# In[12]:


##checking missing values
df.isnull().sum()


# In[13]:


## checking duplicate values
df.duplicated().sum()


# In[14]:


## remove duplicates
df=df.drop_duplicates(keep='first')


# In[15]:


df.duplicated().sum()


# # EDA

# In[16]:


df['target'].value_counts()


# In[17]:


plt.pie(df['target'].value_counts(),labels=['ham','spam'],autopct="%0.2f")
plt.show()


# In[18]:


## Data is imbalanced
import nltk
get_ipython().system('pip install nltk')


# In[19]:


nltk.download('punkt')


# In[ ]:


df['text']


# In[21]:


df['num_characters'] = df['text'].apply(len)


# In[22]:


df.head()


# In[23]:


## num of words
df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[24]:


df['num_words']


# In[25]:


df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[26]:


df['num_sentences']


# In[27]:


df[['num_characters','num_words','num_sentences']].describe()


# In[28]:


## for ham messages
df[df['target'] == 0][['num_characters','num_words','num_sentences']].describe()


# In[29]:


## for spam messages
df[df['target'] == 1][['num_characters','num_words','num_sentences']].describe()


# In[30]:


## comparing the above two types using histogram
plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_characters'])
sns.histplot(df[df['target'] == 1]['num_characters'], color='red')


# In[31]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_words'])
sns.histplot(df[df['target'] == 1]['num_words'], color='red')


# In[32]:


sns.pairplot(df,hue='target')


# In[33]:


sns.heatmap(df.corr(),annot=True)


# DATA PREPROCESSING
# 1) Lower Case
# 2) Tokenization
# 3) Removing Special Characters
# 4) Removing Stop Words and Punctuation
# 5) Stemming
# 

# In[34]:


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y=[]
    
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
            
        
    
            
    return " ".join(y)


# In[81]:


transform_text('i loved the youtube Lectures on Machine Learning. How about you??')


# In[36]:


## removing special characters
df['text'][8]


# In[37]:


##removing stopwords and punctuation
from nltk.corpus import stopwords
stopwords.words('english')


# In[38]:


##removing punctuation
import string
string.punctuation


# In[39]:


## stemming
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
ps.stem('dancing')


# In[40]:


df['transformed_text'] = df['text'].apply(transform_text)


# In[41]:


df.head()


# In[42]:


from wordcloud import WordCloud
wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')


# In[43]:


spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))


# In[44]:


plt.figure(figsize=(12,6))
plt.imshow(spam_wc)


# In[45]:


ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))


# In[46]:


plt.figure(figsize=(12,6))
plt.imshow(ham_wc)


# In[47]:


## top spam words
spam_corpus=[]
for msg in df[df['target'] == 1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[48]:


len(spam_corpus)


# In[49]:


from collections import Counter
sns.barplot(pd.DataFrame(Counter(spam_corpus).most_common(30))[0],pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation = 'vertical')
plt.show()


# In[50]:


## top ham words
ham_corpus=[]
for msg in df[df['target'] == 0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


# In[51]:


sns.barplot(pd.DataFrame(Counter(ham_corpus).most_common(30))[0],pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation = 'vertical')
plt.show()


# In[52]:


len(ham_corpus)


# ## Model Building

# In[53]:


##Text Vectorization
## Using bag of words
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer()


# In[54]:


X = cv.fit_transform(df['transformed_text']).toarray()


# In[55]:


X


# In[56]:


y = df['target'].values


# In[57]:


y


# In[58]:


from sklearn.model_selection import train_test_split


# In[59]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 2)


# In[60]:


##using naive bayes
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix


# In[61]:


gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()


# In[62]:


gnb.fit(X_train,y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[63]:


mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# In[64]:


bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# In[65]:


##using tfidf method
A = tfidf.fit_transform(df['transformed_text']).toarray()


# In[66]:


A_train,A_test,y_train,y_test = train_test_split(A,y,test_size = 0.2,random_state = 2)


# In[67]:


gnb.fit(A_train,y_train)
y_pred1 = gnb.predict(A_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[68]:


mnb.fit(A_train,y_train)
y_pred2 = mnb.predict(A_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# In[69]:


bnb.fit(A_train,y_train)
y_pred3 = bnb.predict(A_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# In[70]:


## TFIDF--> MNB


# In[71]:


##Model Improvement 
##1) Use max features


# In[72]:


tfidff = TfidfVectorizer(max_features=3000)


# In[73]:


C = tfidff.fit_transform(df['transformed_text']).toarray()


# In[74]:


C


# In[75]:


C_train,C_test,y_train,y_test = train_test_split(C,y,test_size = 0.2,random_state = 2)


# In[76]:


bnb.fit(C_train,y_train)
y_pred3 = bnb.predict(C_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# In[80]:


import pickle
pickle.dump(tfidff,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))


# In[78]:


mnb.fit(C_train,y_train)
y_pred3 = mnb.predict(C_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# In[ ]:




