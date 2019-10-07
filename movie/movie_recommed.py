from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np


def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]
def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]


# define a sample text
text= ["London Paris London India ","India Paris Paris London","London India"]

#cv is the object of class CountVectorizer
cv = CountVectorizer()

#a variable to stored the vector of textt_matrix = cv.fit_transform(text)

# to array will convert matrix into an array

print((count_matrix).toarray())

#calculating similarity between elements
similarity_scores = cosine_similarity(count_matrix)
print(similarity_scores)



#read dataset of movies 
df = pd.read_csv("movie_dataset.csv")

#data processing(set col. limit to 23 and 24 is bound)

df = df.iloc[ : ,0:24]

print(df.columns)

#Select some featurs(Properties)

features = ['keywords','cast','genres','director']


#data preprocessing step to remove "NA"
#Fill "NA" with blank space.
#add or delete df[feature]=df.iloc[feature].fillna(' ')

for feature in features:
    df[feature] = df[feature].fillna(' ')
    

#Create a column in dataframe to combine features.
    
def combine_features(row):
    return row['keywords'] + " " + row['cast'] + " " + row['genres'] +  " " + row['director']


df["combine_features"] = df.apply(combine_features,axis=1)

print(df["combine_features"].head())

count_matrix = cv.fit_transform(df["combine_features"])

cosin_sim = cosine_similarity(count_matrix)

print((count_matrix).toarray())

movie_user_likes = "Avengers: Age of Ultron"
# movie_user_likes = "Avatar"


#get index of movies 

movie_index = get_index_from_title(movie_user_likes)

#list of tuples of similar movies
similar_movies = list(enumerate(cosin_sim[int(movie_index)]))

#sort the tuple


sorted_similar_movies =  sorted(similar_movies,key= lambda x:x[1],reverse=True)

print(sorted_similar_movies)

#print the title
 #i=0
 
 for movie in sorted_similar_movies:
     #call titles from movie indices.
     print(get_title_from_index(movie[0]))
     i=i+1
     if i>5:
         break