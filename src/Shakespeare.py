
# coding: utf-8

# # EECS 731 - To be, or not to be
# #### Homework 2
# 
# ## Introduction
# Shakespeare is well known for his plays. In all the created plays, multiple characters exist with various lines. Using text and line data from Shakespeare's plays, can the character be classified? To further breakdown the problem, certain characters appear in only certain plays. This provides better dependency for character detection. Furthermore, characters only appear in certain scenes. By parsing the act/scene line, it's possible to take this into consideration. Lastly, certain terms are more frequent among certain players compared to other players. With these features, it is possible to correctly classify.

# In[1]:


#Import libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn import preprocessing
from sklearn import naive_bayes

#Import dataset
srcSet=pd.read_csv('../data/external/Shakespeare_data.csv')


# ## Cleaning Dataset
# 
# First, the dataset needs cleaning of text without associated characters as this is not classifiable unless stating NaN. However, this is irrelevant for the assignment. 
# 
# Next, the data line is removed as indexing accounts for this.
# 
# Lastly, all empty values for ActSceneLine are removed as this is used to determine character. Furthermore, the empty values tend to indicate an "Entrance" or "Exit".

# In[2]:


#Remove NaN for Player
srcSet=srcSet.dropna(subset=['Player'])

#Remove data scene line as indexing accounts for such
srcSet=srcSet.iloc[:,2:6]

#Remove NaN for ActSceneLine
srcSet=srcSet.dropna(subset=['ActSceneLine'])


# ## Transforming 
# In order to generalize the acts and scenes for better classification, the act-scene-line must be parsed. Then the line and original act-scene-line are dropped as they are irrelevant.

# In[3]:


#Copy column for manipulation
actSceneSplit=srcSet['ActSceneLine'].copy()

#Split the column string data into Act, Scene and Line numbers
actSceneSplit=actSceneSplit.str.split('.', expand=True)

#Rename columns to appropriate label
actSceneSplit.columns=['Act', 'Scene', 'Line']

#Merge with existing dataframe
srcSet=srcSet.join(actSceneSplit)

#Drop ActSceneLine and Line
srcSet=srcSet.drop(columns=['ActSceneLine', 'Line'])


# ## Data Modeling
# Before classification of characters in Shakespeare's play, the data needs additional value and formality. Since player's talking in text lines, multiple values can be extracted. First, the text is broken up into words for each player. Second, the frequency of words for each player is determined by counting total words and amount of repeats. Lastly, the words require transformation from string to numerical values. This is accomplished with hashing. 
# 
# This data is then exported to a csv for internal data usage.

# In[4]:


#create dictionary for each character for learning

#extract player name and line
doc=srcSet

#Alter line to array of words
doc['words']=doc['PlayerLine'].str.strip().str.split('[^A-Za-z0-9\']+')

#Drop the player lines
doc=doc.drop(columns=['PlayerLine'])


#Create new dataframe focused on each word with each Player
rows=list()
for row in doc[['Player','PlayerLinenumber','Act','Scene','words']].iterrows():
    r=row[1]
    for word in r.words:
        rows.append((r['Player'],r['PlayerLinenumber'],r['Act'],r['Scene'], word))

wordDoc=pd.DataFrame(rows, columns=['Player', 'PlayerLinenumber','Act','Scene','Words'])

#Remove all empty values
wordDoc=wordDoc[wordDoc['Words'].str.len() > 0] 


#Start normalizing by counting the amount of same words
wordCount=wordDoc.groupby(['Player','PlayerLinenumber','Act','Scene']).Words.value_counts().to_frame().rename(columns={'Words':'wc'})


#Equate the counts of words for each player
word_total=wordCount.groupby(level=0).sum().rename(columns={'wc':'nt'})

#Calculate ratio of total words and single words for each Player
perPlayerrate=wordCount.join(word_total)

#Place ratio into corpus by merging on Player
perPlayerrate['tf']=perPlayerrate.wc/perPlayerrate.nt

#output to internal data
perPlayerrate.to_csv('../data/internal/textTF.csv')

randomForestData=pd.read_csv('../data/internal/textTF.csv')
randomForestData['Words']=randomForestData.Words.apply(hash)


# ## Classification Models
# Due to the numerical values of columns, I chose to implement Random Forests and SVMs.
# 
# Before inputting the feature space, the data transformed and modelled requires a few more alterations. First, data needs dropping of word count and repetition of a player as these are equated in column TF. Furthermore, the data is split into 80/20/20 for training, testing and validation.
# 
# #### Random Forest
# The Random Forest Classifier is implemented. The features and label data is trained then the error is determined by how many labels were not correctly predicted.

# In[5]:


#----------Further transform data for machine learning usability------
#import data
randomForestData=pd.read_csv('../data/internal/textTF.csv')

#Drop word count and total word count
randomForestData=randomForestData.drop(columns=['wc','nt','Words'])

#Labelled data 
label=randomForestData['Player']

#Drop labelled data and state features
features=randomForestData.drop(columns=['Player'],axis=1)

#For plotting, column list is saved
feature_list=list(randomForestData.columns)

#Normalize the finished dataset before converting to array
mini=randomForestData['PlayerLinenumber'].min()
maxi=randomForestData['PlayerLinenumber'].max()

randomForestData['PlayerLinenumber']=randomForestData['PlayerLinenumber'].apply(lambda x: (x-mini)/(maxi-mini))

mini=randomForestData['Act'].min()
maxi=randomForestData['Act'].max()

randomForestData['Act']=randomForestData['Act'].apply(lambda x: (x-mini)/(maxi-mini))

mini=randomForestData['Scene'].min()
maxi=randomForestData['Scene'].max()

randomForestData['Scene']=randomForestData['Scene'].apply(lambda x: (x-mini)/(maxi-mini))

#Convert to numpy array
features=np.array(features)

#---------Training/Testing/Validation--------
x, x_test, y, y_test = train_test_split(features,label,test_size=0.1,train_size=0.9)
x_train, x_val, y_train, y_val = train_test_split(x,y,test_size = 0.15,train_size =0.85)


# In[6]:


#------------Random Forest Implementation----------
#-------Data Counts------
#527979 - train
#65998 - test
#65997 - validate

#----------Train---------
rf=RFC()

rf.fit(x_train, y_train)

#--------Testing Predictions and Errors----------
predictions=rf.predict(x_test)

#setup both sets
y_test=y_test.reset_index()
y_test=y_test.drop(columns=['index'])

pred=pd.DataFrame(predictions)

mergd=pred.join(y_test)
mergd.columns=['Predict', 'Actual']

incorrect=0
total=0

#calculate the errors by determining how many correct predictions
for index,row in mergd.iterrows():
    row
    if row['Predict'] != row['Actual']:
        incorrect+=1
    total+=1
    
correct=total-incorrect

#Outputting accuracy of model
if total!=0:
    print('Accuracy for Testing:', round(correct/float(total),2))
else:
    print('No items...')


# In[7]:


#--------Validating training and testing data----------
predictions=rf.predict(x_val)

#setup both sets
y_val=y_val.reset_index()
y_val=y_val.drop(columns=['index'])

pred=pd.DataFrame(predictions)

mergd=pred.join(y_val)
mergd.columns=['Predict', 'Actual']

incorrect=0
total=0

#calculate the errors by determining how many correct predictions
for index,row in mergd.iterrows():
    row
    if row['Predict'] != row['Actual']:
        incorrect+=1
    total+=1
    
correct=total-incorrect

#Outputting accuracy of model
if total!=0:
    print('Accuracy for Validation:', round(correct/float(total),2))
else:
    print('No items...')


# #### Naive Bayes
# Naive Bayes is used to multi-class classification and is good for test/categorical classification. This algorithm works well with data occurrence counts. For this data, the only occurrence is through TF.

# In[8]:


naive = naive_bayes.MultinomialNB()
naive.fit(x_train, y_train)


# In[9]:


#--------Testing Predictions and Errors----------
predictions=naive.predict(x_test)

#setup both sets
y_test=y_test.reset_index()
y_test=y_test.drop(columns=['index'])

pred=pd.DataFrame(predictions)

mergd=pred.join(y_test)
mergd.columns=['Predict', 'Actual']

incorrect=0
total=0

#calculate the errors by determining how many correct predictions
for index,row in mergd.iterrows():
    row
    if row['Predict'] != row['Actual']:
        incorrect+=1
    total+=1
    
correct=total-incorrect

#Outputting accuracy of model
if total!=0:
    print('Accuracy:', round(correct/float(total),2))
else:
    print('No validated data misclassified...')


# In[10]:


#--------Validating training and testing data----------
predictions=naive.predict(x_val)

#setup both sets
y_val=y_val.reset_index()
y_val=y_val.drop(columns=['index'])

pred=pd.DataFrame(predictions)

mergd=pred.join(y_val)
mergd.columns=['Predict', 'Actual']

incorrect=0
total=0

#calculate the errors by determining how many correct predictions
for index,row in mergd.iterrows():
    row
    if row['Predict'] != row['Actual']:
        incorrect+=1
    total+=1
    
correct=total-incorrect

if total!=0:
    print('Accuracy for Validation:', round(correct/float(total),2))
else:
    print('No items...')


# ## Analysis
# The two classification models used were Random Forest Generator and Naive Bayes. 
# 
# #### Random Forest
# To measure the success of this model, the accuracy is the true positives (true classification) compared to the entire dataset. This accuracy is measured to be 0.97. This is good as 1 is 100% rate. Due to the feature set, this makes sense as the data is categorized based on features such as when players appear in scenes and acts as well as frequency of certain teminology.
# 
# #### Naive Bayes
# Measurement of success is the same as random forest. The accuracy is determined based on correctness of classification. However, unlike random forests, the accuracy calculated is 0.02. This is due to the feature type. This algorithm does well with occurence features such as term frequency. This is a feature in the data set; however, the acts and scene columns are not based on frequencies. To alleviate this in the future, it's possible to create each scene and act as a separate column and record the occurences of players in each.
# 

# In[12]:


#Create table to compare results

output={'Algorithm': ['Random Forest','Naive Bayes'], 'Accuracy': [0.98, 0.02]}
output=pd.DataFrame(data=output)
output


# ## Previous Trials
# ------------------------------------ 
# 
# #### SVM
# The Support Vector Machine is the second classification model. Took too long to train.
# 
# Below is all the programming done for this:
# 
# ---------------------------------------------------------
# from sklearn import svm
# 
# #----------Train SVM---------
# #Standardize the data
# x_scale=preprocessing.scale(x_train)
# x_teststd=preprocessing.scale(x_test)
# x_valstd=preprocessing.scale(x_val)
# 
# clf=svm.SVC(kernel='sigmoid', verbose=2, tol=0.1, )
# 
# clf.fit(x_train, y_train)
# 
# predictions=clf.predict(x_test)
# 
# #setup both sets
# y_test=y_test.reset_index()
# y_test=y_test.drop(columns=['index'])
# 
# pred=pd.DataFrame(predictions)
# 
# mergd=pred.join(y_test)
# mergd.columns=['Predict', 'Actual']
# 
# #### Logistic Regression
# Attempted to use logistic regression, but never stopped even on convergence....
# 
# ---------------------------------------------------------
# #Standardize the data
# x_scale=preprocessing.scale(x_train)
# x_teststd=preprocessing.scale(x_test)
# x_valstd=preprocessing.scale(x_val)
# 
# logistReg=LogisticRegression(max_iter=40,random_state=0, solver='sag', verbose=3)
# logistReg.fit(x_scale, y_train)
# 
# #--------Testing Predictions and Errors----------
# predictions=logistReg.predict(x_teststd)
# 
# #setup both sets
# y_test=y_test.reset_index()
# y_test=y_test.drop(columns=['index'])
# 
# pred=pd.DataFrame(predictions)
# 
# mergd=pred.join(y_test)
# mergd.columns=['Predict', 'Actual']
# 
# etc.....
# 
# -------------------------------
# ## Conclusion
# For determining players based on lines in a play, the best features include the scene, act, player number and term frequencies. In terms of model, since these features consist of frequency data and discrete values, random forests work best especially considering the amount of rows. Naive Bayes may work, but the current features need readjusting to all frequency data instead of a combination of both. 

# ----------------
# 
# ## Resources
# 
# #### Text Analysis in Pandas
# https://sigdelta.com/blog/text-analysis-in-pandas/
# 
# #### Using Random Forests
# https://towardsdatascience.com/random-forest-in-python-24d0893d51c0 
# 
# #### To split train/test/validation to 80/10/10
# https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test/38251213#38251213
# 
# #### For counter to create array of used words
# https://stackoverflow.com/questions/51280327/trying-to-create-a-bag-of-words-of-pandas-df
# 
# #### List difference
# https://stackoverflow.com/questions/6486450/python-compute-list-difference/6486467
# 
# #### Logisitic Regression on Very Large data
# https://chrisalbon.com/machine_learning/logistic_regression/logistic_regression_on_very_large_data/
