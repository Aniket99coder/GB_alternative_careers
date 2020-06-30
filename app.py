import streamlit as st
import pandas as pd

st.title("Hello! Use this to get yourself an alternative career")
#Loading the data
DATA_URL_GB="gb_types.csv"
@st.cache(persist=True) #Cached data no need for loading
def load_data_GB():
    data=pd.read_csv(DATA_URL_GB)    
    return data

DATA_URL_BIGFIVE="global_5_data_32_types.csv"
@st.cache(persist=True) #Cached data no x`xneed for loading
def load_data_BIGFIVE():
    data=pd.read_csv(DATA_URL_BIGFIVE)    
    return data

data_gb = load_data_GB()
big_five_df=load_data_BIGFIVE()

#Creating a menu of personality traits
# Create a list of possible values and multiselect menu with them in it.
menu=data_gb[['Personality GB','Trait','Detail']].copy()[:-1]
TYPES_GB = menu['Personality GB'].unique()
TYPES_GB_SELECTED = st.multiselect('Select Personality Types', TYPES_GB)

# Mask to filter dataframe
mask_types = menu['Personality GB'].isin(TYPES_GB_SELECTED)

types_select= menu[mask_types]

#Writing in the streamlit
st.write(types_select)

# Program to measure similarity between  
# two sentences using cosine similarity. 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

# Concatinating all input types of gb by user 
arc_type_1=""
# Adding all the details of the selected
for i in types_select['Detail']:
	arc_type_1+=i
big_five_keys=big_five_df['Keywords']
index=0
score=[]
name=[]
dict_career={}
dict_n_career={}
for big_five_key in big_five_keys:
    # X = input("Enter fist string: ").lower() 
    # Y = input("Enter second string: ").lower() 
    X =arc_type_1
    Y =big_five_key
    
    X=X.lower()
    Y=Y.lower()
    # tokenization 
    X_list = word_tokenize(X)  
    Y_list = word_tokenize(Y) 

    # sw contains the list of stopwords 
    sw = stopwords.words('english')  
    l1 =[];l2 =[] 

    # remove stop words from string 
    X_set = {w for w in X_list if not w in sw}  
    Y_set = {w for w in Y_list if not w in sw} 

    # form a set containing keywords of both strings  
    rvector = X_set.union(Y_set)  
    for w in rvector: 
        if w in X_set: l1.append(1) # create a vector 
        else: l1.append(0) 
        if w in Y_set: l2.append(1) 
        else: l2.append(0) 
    c = 0

    # cosine formula  
    cosine=0
    for i in range(len(rvector)): 
            c+= l1[i]*l2[i] 
    try :
    	cosine = c / float((sum(l1)*sum(l2))**0.5)
    except ZeroDivisionError:
    	cosine = 0
    score.append(cosine)
    name.append(big_five_df['Type'][index])    
    dict_career[big_five_df['Type'][index]]=big_five_df['Favored Careers'][index]
    dict_n_career[big_five_df['Type'][index]]=big_five_df['Disfavoured Careers'][index]
    index=index+1
x=sorted(zip(score, name), reverse=True)[:8]
	

import math
import numpy as np
#Taking the union of the matching personality

set_f=set()
for i in range(0,2):
    key=x[i][1]
    
    set_1=set(dict_career[key].split(","))    
    set_f=set_f.union(set_1)
    



#Removing the disfavored careers of the similar personality    
for i in range(0,8):    
    key=x[i][1]
    
    if dict_n_career[key] is np.nan:
        set_n_1=set()
    else:
        set_n_1=set(dict_n_career[key].split(","))
        
    set_f=set_f-set_f.intersection(set_n_1)

st.write(set_f)

from collections import Counter
import matplotlib.pyplot as plt 
from wordcloud import WordCloud

def  generate_image():	 
	word_could_dict=Counter(list(set_f))
	wordcloud = WordCloud(width = 2000, height = 2000).generate_from_frequencies(word_could_dict)
	plt.figure(figsize=(50,50))
	plt.imshow(wordcloud)
	plt.axis("off")
	plt.show()
	#plt.savefig('yourfile.png', bbox_inches='tight')	    
	st.pyplot()   

if st.button('Generate Alternative Careers'):
    result = generate_image()
    

