# pehly libraries import kar lain
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# app ke heading
st.write("""
# Explore different ML models and datasets
Daikhty han kon sa best ha in may?""")

# data set k name ek box ma dal kr sidebar pa laga do
dataset_name = st.sidebar.selectbox(
    ' select dataset',('Iris','Breast cancer','Wine')
    )

# or isi k nechy classifier k name ek daby ma dal do
classifier_name = st.sidebar.selectbox('select classifier',('KNN','SVM','Random Forest'))

# ab hum ny ek funcition define karna ha dataset load karny k lye

def get_dataset(dataset_name):
    data = None
    if dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    x = data.data
    y = data.target
    return x,y

# ab is funcition ko bula ly gy or x and y k equal rak ly gy
x,y = get_dataset(dataset_name)

# ab hum is dataset ke shape print kary gy app py
st.write('shape of dataset:',x.shape)
st.write('Number of classes:',len(np.unique(y)))


# next hum differnt classifier k parameter ko user input ma add kary gy

def add_parameter_ui(classifier_name):
    params = dict()   # create empty dictionary
    if classifier_name == 'SVM':
        c = st.sidebar.slider('C',0.01,10.0)
        params['C'] = c # its degree of correct classification
    elif classifier_name == 'KNN':
        k = st.sidebar.slider('K',1,15)
        params['K'] = k  # its number of nearest neighours
    else:
        max_depth = st.sidebar.slider('max_depth',2,15)
        params['max_depth'] = max_depth # depth of every tree that grow in random forest
        n_estimators = st.sidebar.slider('n_estimators',1,100)
        params['n_estimators']  = n_estimators  # number of trees
    return params

# ab is funtion ko call karty hain or params variable k equal rak ly gy
params = add_parameter_ui(classifier_name)


# ab hum classifier banyen gy based on classifier name and params

def get_classifier(classifier_name,params):
    clf = None
    if classifier_name == 'SVM':
        clf = SVC(C=params['C'])
    elif classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
       clf= clf = RandomForestClassifier(n_estimators= params['n_estimators'],
                    max_depth = params['max_depth'], random_state = 1234)
    return clf

# ab hum is funtion ko bula len gy or clf k equal rak ly gy
clf = get_classifier(classifier_name, params)


# ab hum data ko split kar lengy  train or test data ma 80/20 ratio ma
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1234)

# ab humy apny classifier k training krni ha
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

# model ke accurcy check kry gy or os ko app ma print kr lengy
acc = accuracy_score(y_test,y_pred)
st.write(f'classifier = {classifier_name}')
st.write(f'Accuracy =',acc)


### plot dataset ###
# ab hum  apny saray k sary features ko 2 dimentional  plot pa draw  kar dayn gy using pca
pca = PCA(2)
x_projected = pca.fit_transform(x)

# ab hum apny data ko 0 and 1 dimension ma slice kar dengy

x1 = x_projected[ :,0]
x2 = x_projected[ :,1]

fig = plt.figure()
plt.scatter(x1,x2,c=y,alpha = 0.8,cmap = 'viridis') 

plt.xlabel('principal component 1')
plt.ylabel('principal component 2')
plt.colorbar()

# plt show
st.pyplot(fig)





