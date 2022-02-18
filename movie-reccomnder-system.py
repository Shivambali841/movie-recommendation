#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies.head()


# In[4]:


credits.head(1)


# In[5]:


movies=movies.merge(credits,on='title') #merge the data on the basis of title 


# In[6]:


movies.shape


# In[7]:


credits.shape


# In[8]:


movies.head(1)


# In[9]:


# genre
#id
# keywords
# title
# overview
# cast
# crew
movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[10]:


2+2


# In[11]:


movies.head()


# #### Now next strategy is to create tags for that, I will merge overview,gernes,keywords,cast and crew and take out the tags

# In[12]:


movies.isnull().sum()


# In[13]:


movies.dropna(inplace=True) # since 3 is a small number


# In[14]:


movies.duplicated().sum() # To check if there is any duplicate data


# In[15]:


movies.iloc[0].genres


# In[16]:


import ast


# In[17]:


def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[18]:


movies['genres']=movies['genres'].apply(convert)


# In[19]:


movies.head()


# In[20]:


movies['keywords']=movies['keywords'].apply(convert)


# In[21]:


movies.head()


# In[22]:


movies['cast'][0]


# In[23]:


def convert3(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[24]:


movies['cast']=movies['cast'].apply(convert3)


# In[25]:


movies.head()


# In[26]:


movies['crew'][0]


# In[27]:


def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break
    return L


# In[28]:


movies['crew']=movies['crew'].apply(fetch_director)


# In[29]:


movies.head()


# #### Change the overview from STRING TO list so it contatinates with other variables

# In[30]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[31]:


movies['overview']


# ### Now we concatinates all the list and convert that list to the string and that will make a huge paragraph, tags

# In[32]:


movies.head()


# ### now deletes the space between the name 

# In[33]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])


# In[34]:


movies.head()


# In[35]:


movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[36]:


movies.head(1)


# In[37]:


new_df=movies[['movie_id','title','tags']]


# In[38]:


new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))


# In[39]:


new_df['tags'][0]


# In[40]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower()) # convert into lower case.since its preffered


# In[41]:


new_df.head()


# ## TEXT VECTORIZATION

# ### BAG OF WORDS

# In[42]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')


# In[43]:


vectors=cv.fit_transform(new_df['tags']).toarray() # here we need to change it into numpy array


# In[44]:


vectors.shape


# In[45]:


vectors[0]


# In[46]:


cv.get_feature_names() 


# ### STEMMING 

# #### To get rid of the similar words. By changing into the root words 

# In[47]:


import nltk


# In[48]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[49]:


def stem(text):
    y=[]
    
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)


# In[50]:


new_df['tags']=new_df['tags'].apply(stem)


# In[51]:


cv.get_feature_names()


# #### Evaluating Cosine Distance between every movie,Since it is a high dimensional data we will use cosine deiatance to calcualte the similarities

# In[52]:


from sklearn.metrics.pairwise import cosine_similarity


# In[53]:


similarity=cosine_similarity(vectors)


# In[54]:


similarity[0]


# In[55]:


def recommend(movie):
    movie_index=new_df[new_df['title']==movie].index[0] #fetch the index
    distances=similarity[movie_index] #calculate distance
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    #enumerate to mantain the index and then sort in top 5
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
        
    


# In[56]:


recommend('The Dark Knight')


# In[57]:


import pickle
pickle.dump(new_df,open('movies.pkl','wb'))


# In[58]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))


# In[59]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




