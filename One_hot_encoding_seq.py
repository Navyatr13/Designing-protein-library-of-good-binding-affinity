#!/usr/bin/env python
# coding: utf-8

# In[198]:


# One hot encoding for the sequences
# August 5th 


import pandas as pd 
import numpy as np
import math
from sklearn import preprocessing
from functools import partial
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors
from matplotlib.colors import LogNorm
np.warnings.filterwarnings('ignore')

from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def encode_paratope_OH(enc,paratope,axis):
    paratope=np.array(list(paratope))
    one_encode=enc.transform(paratope.reshape(-1,1))
    return one_encode.flatten()

def main():

    AAlist=np.array(list("ACDEFGHIKLMNPQRSTVWXYZ_-"))
    enc_OH=preprocessing.OneHotEncoder(sparse=False)
    enc_OH.fit(AAlist.reshape(-1,1))
    clust1 =  np.array(np.zeros((5000,5000)))
    clust1 = clust1.tolist()
    counter = 0
    print("AAlist")
    print(AAlist)
    encode_OH=partial(encode_paratope_OH,enc_OH)
    print("encode_paratope_OH")
    #print(encode_OH)
    
    df = pd.read_csv('Seq_Fitness_example.csv')
    df.loc[:,'One_Hot'] = df['Sequence'].apply(encode_OH,axis = 1)
    x_a=df.loc[:,'One_Hot'].values.tolist()
    for i in range(len(x_a)):
        x_a[i]=x_a[i].tolist()
    x_a=np.array(x_a)
    x_aa = np.asarray(x_a)
    x_train = x_aa[0:300,0]
    dff = np.asarray(df)
    data_top = df.head()
    print(list(df))
    return dff

if __name__ == '__main__':
    result = main()
    import numpy as np

    #from sklearn.cross_validation import train_test_split
    from sklearn.model_selection import train_test_split

    # create design matrix X and target vector y
    X = np.array(list(result[:,2])) # end index is exclusive
    y = np.array(list(result[:,1]))   # another way of indexing a pandas df

    # split into train and test
    X_train,     X_test,     y_train,     y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[199]:


print(X_train)


# In[200]:


print(result[:,0])


# In[22]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# instantiate learning model (k = 3)
knn = KNeighborsClassifier(n_neighbors=500)
# fitting the model
knn.fit(X_train, y_train)

# predict the response
pred = knn.predict(X_test)
print(len(pred))

# evaluate accuracy
print("accuracy: {}".format(accuracy_score(y_test, pred)))
confusion_matrix(y_test, pred)


# In[36]:


y_test


# In[35]:


print((pred))


# # K-nearest neighbor algorithm

# In[207]:



# Selecting the best K value in the range of 10: 10 clusters:
# creating odd list of K for KNN
neighbors = list(range(1, 2, 2))
from sklearn.model_selection import cross_val_score
# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    print(scores)
    cv_scores.append(scores.mean())


# # K-nearest neighbor algorithm for clusters ranging 300

# In[17]:


# creating odd list of K for KNN
neighbors = list(range(1,300, 2))
from sklearn.model_selection import cross_val_score
get_ipython().run_line_magic('matplotlib', 'inline')

# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())
import matplotlib.pyplot as plt
# changing to misclassification error
mse = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[mse.index(min(mse))]
print("The optimal number of neighbors is {}".format(optimal_k))

# plot misclassification error vs k
plt.plot(neighbors, mse)
plt.xlabel("Number of Neighbors K")
plt.ylabel("Misclassification Error")
plt.show()


# In[210]:


# creating odd list of K for KNN
neighbors = list(range(1,50, 2))
from sklearn.model_selection import cross_val_score
get_ipython().run_line_magic('matplotlib', 'inline')

# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())
import matplotlib.pyplot as plt
# changing to misclassification error
mse = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[mse.index(min(mse))]
print("The optimal number of neighbors is {}".format(optimal_k))

# plot misclassification error vs k
plt.plot(neighbors, mse)
plt.xlabel("Number of Neighbors K")
plt.ylabel("Misclassification Error")
plt.show()


# # KNN based on the best neighbor prediction

# In[212]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=129)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[232]:


y_pred[0:100]


# In[233]:


y_test[0:100]


# In[211]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=39)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# # SVM with linear kernel

# In[124]:


from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# # SVM with kernel-poly

# In[126]:


from sklearn.svm import SVC
svclassifier = SVC(kernel='poly')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# # SVM with kernel-rbf

# In[127]:


from sklearn.svm import SVC
svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# # SVM with kernel- sigmoid

# In[128]:


from sklearn.svm import SVC
svclassifier = SVC(kernel='sigmoid')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[72]:


print(len(y_test))


# # Data preparing for NN & CNN

# In[131]:


# K-means for training data
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=8)
kmeans.fit(X_train)
y_kmeans = kmeans.predict(X_train)


# In[238]:


y_kmeans[0:50]


# In[148]:


# K-means for test data
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=8)
kmeans.fit(X_test)
y_kmeans_test = kmeans.predict(X_test)


# In[213]:


print(len(y_kmeans), len(X_train))#[0:1000]


# In[134]:


# applying t-SNE for visualising the clustered data

from sklearn.manifold import TSNE
import time
import numpy as np

RS = 123
tsne = TSNE(random_state=RS).fit_transform(X_train)
 
class1=[]
class2=[]
class3=[]
class4=[]
class5=[]
class6=[]
class7=[]
class8=[]


count1=0
count2=0
count3=0
count4=0
count5=0
count6=0
count7=0
count8=0


for i in range(tsne.shape[0]):
    if np.array(y_kmeans[i])==0:
        class1.extend(tsne[i])
        count1=count1+1
    elif np.array(y_kmeans[i])==1:
        class2.extend(tsne[i])
        count2=count2+1
    elif np.array(y_kmeans[i])==2:
        class3.extend(tsne[i])
        count3=count3+1
    elif np.array(y_kmeans[i])==3:
        class4.extend(tsne[i])
        count4=count4+1
    elif np.array(y_kmeans[i])==4:
        class5.extend(tsne[i])
        count5=count5+1
    elif np.array(y_kmeans[i])==5:
        class6.extend(tsne[i])
        count6=count6+1
    elif np.array(y_kmeans[i])==6:
        class7.extend(tsne[i])
        count7=count7+1
    elif np.array(y_kmeans[i])==7:
        class8.extend(tsne[i])
        count8=count8+1
    ''''elif np.array(y_kmeans[i])==8:
        class9.extend(tsne[i])
        count9=count9+1
    elif np.array(y_kmeans[i])==9:
        class10.extend(tsne[i])
        count10=count10+1
    elif np.array(y_kmeans[i])==10:
        class11.extend(tsne[i])
        count11=count11+1
    elif np.array(y_kmeans[i])==11:
        class12.extend(tsne[i])
        count12=count12+1
    elif np.array(y_kmeans[i])==12:
        class13.extend(tsne[i])
        count13=count13+1
    elif np.array(y_kmeans[i])==13:
        class14.extend(tsne[i])
        count14=count14+1
    elif np.array(y_kmeans[i])==14:
        class15.extend(tsne[i])
        count15=count15+1'''


# In[214]:


print(count1,count2,count3,count4,count5,count6,count7,count8)


# In[138]:


class1=np.reshape(np.array(class1),(count1,2))
class2=np.reshape(np.array(class2),(count2,2))
class3=np.reshape(np.array(class3),(count3,2))
class4=np.reshape(np.array(class4),(count4,2))
class5=np.reshape(np.array(class5),(count5,2))

class6=np.reshape(np.array(class6),(count6,2))
class7=np.reshape(np.array(class7),(count7,2))
class8=np.reshape(np.array(class8),(count8,2))



# In[139]:


import seaborn as sns
x=sns.palplot(sns.color_palette("husl", 19))
f = plt.figure(figsize=(8, 8))
sns.set_palette(x)
ax = plt.subplot(aspect='equal')
ax.scatter(class1[:,0],class1[:,1],label="BEGAN",color="r")
ax.scatter(class2[:,0],class2[:,1],label="DCGAN",color="b")
ax.scatter(class3[:,0],class3[:,1],label="DFCVAE",color="g")
ax.scatter(class4[:,0],class4[:,1],label="FACTOR_VAE",color="c")
ax.scatter(class5[:,0],class5[:,1],label="MAGAN",color="m")
ax.scatter(class6[:,0],class6[:,1],label="MRGAN",color="y")
ax.scatter(class7[:,0],class7[:,1],label="SAGAN",color="crimson")
ax.scatter(class8[:,0],class8[:,1],label="SNGAN",color="dimgray")



# # Multi-layer Perceptron

# In[216]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from keras.models import Sequential
from keras.layers import Dense, Embedding

model = Sequential()
model.add(Dense(90, input_dim=X_train.shape[1], activation='relu'))#ernel_initializer='he_normal'))
model.add(Dense(1,input_dim=90,activation='sigmoid'))
# compile the keras model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X_train, y_kmeans, epochs=100,batch_size=32, verbose=2)
# evaluate the keras model
score = model.evaluate(X_test, y_kmeans_test)
print("")
print("Accuracy = " + format(score[1]*100, '.2f') + "%")   # 92.62%


# In[236]:


y_kmeans_test[0:100]


# # CNN

# In[217]:


from keras.layers import Input, Dense, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout

X_train = X_train.reshape(3147,720,1)
#print(X_train.shape[1])
X_test = X_test.reshape(1350,720,1)

model_scratch = Sequential()
model_scratch.add(Conv1D(64, 3, activation='relu',input_shape = X_train.shape[1:]))
model_scratch.add(MaxPooling1D(pool_size=2))
model_scratch.add(Dropout(0.25))
 
model_scratch.add(Conv1D(64, 3, activation='relu'))
model_scratch.add(MaxPooling1D(pool_size=2))
model_scratch.add(Dropout(0.25))
 
model_scratch.add(Conv1D(64, 3, activation='relu'))
model_scratch.add(MaxPooling1D(pool_size=2))
model_scratch.add(Dropout(0.25))
 
model_scratch.add(Conv1D(128, 3, activation='relu'))
model_scratch.add(MaxPooling1D(pool_size=2))
model_scratch.add(Dropout(0.25))
 
model_scratch.add(GlobalAveragePooling1D())
model_scratch.add(Dense(64, activation='relu'))
model_scratch.add(Dense(1))
model_scratch.summary()
model_scratch.compile(loss='mean_squared_error', optimizer='adam',
              metrics=['accuracy'])
 
#Fitting the model on the train data and labels.
history = model_scratch.fit(X_train, y_kmeans,
          batch_size=32, epochs=100,
          verbose=1,# callbacks=[checkpointer],
          validation_data=(X_test, y_kmeans_test), shuffle=True)


# In[224]:


X_test = X_test.reshape(1350,720)
scores = model.evaluate(X_test, y_kmeans_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[ ]:




