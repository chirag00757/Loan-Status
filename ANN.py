#Artificial Neural Network

#Installing Theano
#pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

#Installing Tensorflow
#Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.11/get_started

#Installing Keras
#pip install --upgrade keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

loan = pd.read_csv("train.csv")

loan.apply(lambda x:sum(x.isnull()),axis=0)

loan['Gender'].isnull().sum()
loan['Gender'].value_counts()
loan['Gender'].fillna('Male',inplace=True)



#Filling Married
loan['Married'].isnull().sum()
loan['Married'].value_counts()
loan['Married'].fillna('Yes',inplace=True)

#Filling Dependent
loan['Dependents'].isnull().sum()
loan['Dependents'].value_counts()
loan['Dependents'].fillna('2',inplace=True)


#Filling Self_Employee
loan['Self_Employed'].isnull().sum()
loan['Self_Employed'].value_counts()
loan['Self_Employed'] = loan['Self_Employed'].map({"Yes":1,"No":0})

loan.loc[ (pd.isnull(loan['Self_Employed'])) & (loan['Loan_Status'] == 'Y'), 'Self_Employed'] = 0
loan.loc[ (pd.isnull(loan['Self_Employed'])) & (loan['Loan_Status'] == 'N'), 'Self_Employed'] = 1

#Filling Loan_Amount
loan['LoanAmount'].isnull().sum()
loan['LoanAmount'].value_counts()
impute_grps = loan.pivot_table(values=["LoanAmount"], index=["Education","Self_Employed"], aggfunc=np.mean)
print(impute_grps)

for i,row in loan.loc[loan['LoanAmount'].isnull(),:].iterrows():
  ind = tuple([row['Education'],row['Self_Employed']])
  loan.loc[i,'LoanAmount'] = impute_grps.loc[ind].values[0]
  

#Filling Loan_Amount_Term
loan['Loan_Amount_Term'].isnull().sum()
loan['Loan_Amount_Term'].value_counts()
loan['Loan_Amount_Term'].fillna(360,inplace=True)

loan['Loan_Status'] = loan['Loan_Status'].map(lambda x: 1 if x=='Y' else 0)

loan['Credit_History'].fillna(loan['Loan_Status'], inplace = True)


loan['TotalIncome'] = loan['ApplicantIncome'] + loan['CoapplicantIncome']

X = loan.iloc[:, [1,2,4,5,8,9,10,11,13]].values
y = loan.iloc[:, 12].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x_1 = LabelEncoder()
X[:,0] = labelencoder_x_1.fit_transform(X[:,0])
labelencoder_x_2 = LabelEncoder()
X[:,1] = labelencoder_x_2.fit_transform(X[:,1])
labelencoder_x_3 = LabelEncoder()
X[:,2] = labelencoder_x_3.fit_transform(X[:,2])
labelencoder_x_4 = LabelEncoder()
X[:,7] = labelencoder_x_4.fit_transform(X[:,7])

onehotencoder = OneHotEncoder(categorical_features=[7])
X = onehotencoder.fit_transform(X).toarray()



#Remove one dummy variable
X = X[: , 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train , y_test = train_test_split(X,y,test_size = 0.25,random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Part 2 Import keras library and packages

import theano
import tensorflow
import keras


    
from keras.models import Sequential
from keras.layers import Dense

#Initialising the ANN
classifier = Sequential()

#adding the input  layer and the first hidden layer
classifier.add(Dense(output_dim = 5,init= 'uniform',activation='relu',input_dim= 10))

#Adding the second hidden layer
classifier.add(Dense(output_dim = 5,init= 'uniform',activation='relu'))

#Adding the output layer
classifier.add(Dense(output_dim = 1,init= 'uniform',activation='sigmoid'))

#Compiling The ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#Fitting ANN to training set
classifier.fit(X_train,y_train,batch_size=10,nb_epoch=50)


#Part 3  making the prediction and evaluating the model
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#Acuuracy
import numpy as np
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)*100
