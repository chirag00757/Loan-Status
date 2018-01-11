# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 22:44:20 2017

@author: chiku00757
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 13:04:40 2017

@author: chiku00757
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

loan = pd.read_csv("train.csv")
loan.info()

#Finding Misiing value from dataset
loan.apply(lambda x:sum(x.isnull()),axis=0)

#Filling Gender
loan['Gender'].isnull().sum()
loan['Gender'].value_counts()
loan['Gender'].fillna('Male',inplace=True)
#Crosstab
'''
pd.crosstab(loan["Gender"],loan["Loan_Status"],margins=True)

def percConvert(ser):
  return ser/float(ser[-1])
pd.crosstab(loan["Gender"],loan["Loan_Status"],margins=True).apply(percConvert, axis=1)
'''
#Filling Married
loan['Married'].isnull().sum()
loan['Married'].value_counts()
loan['Married'].fillna('Yes',inplace=True)
#Crosstab
'''
pd.crosstab(loan["Married"],loan["Loan_Status"],margins=True)

def percConvert(ser):
  return ser/float(ser[-1])
pd.crosstab(loan["Married"],loan["Loan_Status"],margins=True).apply(percConvert, axis=1)
'''
#Filling Dependent


#Crosstab
'''
pd.crosstab(loan["Dependents"],loan["Loan_Status"],margins=True)

def percConvert(ser):
  return ser/float(ser[-1])
pd.crosstab(loan["Dependents"],loan["Loan_Status"],margins=True).apply(percConvert, axis=1)
'''
loan['Dependents'].isnull().sum()
loan['Dependents'].value_counts()
loan['Dependents'].fillna('2',inplace=True)
'''loan.dtypes'''

#Filling Self_Employee
#Crosstab
'''
pd.crosstab(loan["Self_Employed"],loan["Loan_Status"],margins=True)

def percConvert(ser):
  return ser/float(ser[-1])
pd.crosstab(loan["Self_Employed"],loan["Loan_Status"],margins=True).apply(percConvert, axis=1)

'''
loan['Self_Employed'].isnull().sum()
loan['Self_Employed'].value_counts()
loan['Self_Employed'] = loan['Self_Employed'].map({"Yes":1,"No":0})
'''loan['Self_Employed'].fillna('No',inplace=True)'''
loan.loc[ (pd.isnull(loan['Self_Employed'])) & (loan['Loan_Status'] == 'Y'), 'Self_Employed'] = 0
loan.loc[ (pd.isnull(loan['Self_Employed'])) & (loan['Loan_Status'] == 'N'), 'Self_Employed'] = 1

#Filling Loan amount
impute_grps = loan.pivot_table(values=["LoanAmount"], index=["Education","Self_Employed"], aggfunc=np.mean)
print(impute_grps)

for i,row in loan.loc[loan['LoanAmount'].isnull(),:].iterrows():
  ind = tuple([row['Education'],row['Self_Employed']])
  loan.loc[i,'LoanAmount'] = impute_grps.loc[ind].values[0]
  
  #Filling Loan_Amount_Term
loan['Loan_Amount_Term'].isnull().sum()
loan['Loan_Amount_Term'].value_counts()
loan['Loan_Amount_Term'].fillna(360,inplace=True)

  
#Fiinding Null value counts of Credit History
loan['Credit_History'].isnull().sum()
#Counting values of Credit History
loan['Credit_History'].value_counts()
#Filling Credit History
'''loan.loc[ (pd.isnull(loan['Credit_History'])) & (loan['Loan_Status'] == 'Y'), 'Credit_History'] = 1
loan.loc[ (pd.isnull(loan['Credit_History'])) & (loan['Loan_Status'] == 'N'), 'Credit_History'] = 0
'''
loan['Loan_Status'] = loan['Loan_Status'].map(lambda x: 1 if x=='Y' else 0)

loan['Credit_History'].fillna(loan['Loan_Status'], inplace = True)


#Counting values of Credit History
'''loan['Credit_History'].value_counts()'''


#Crosstab
'''
pd.crosstab(loan["Credit_History"],loan["Loan_Status"],margins=True)

def percConvert(ser):
  return ser/float(ser[-1])
pd.crosstab(loan["Credit_History"],loan["Loan_Status"],margins=True).apply(percConvert, axis=1)

'''
'''pivot = loan.pivot_table(index=['Gender','Married','Credit_History'],values='Loan_Status',aggfunc=lambda x:x.map({'Y':1,'N':0}).mean())
'''
#Total Income
loan['TotalIncome'] = loan['ApplicantIncome'] + loan['CoapplicantIncome']

X = loan.iloc[:, [8,10,11,13]].values
y = loan.iloc[:, 12].values

#Lable Encoder for creating categorical vlaue to Numerical value
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_y = LabelEncoder()
y= labelencoder_y.fit_transform(y)

labelencoder_x_1 = LabelEncoder()
X[:,2] = labelencoder_x_1.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features=[2])
X = onehotencoder.fit_transform(X).toarray()
X = X[: , 1:]
'''
labelencoder_x_2 = LabelEncoder()
X[:,5] = labelencoder_x_2.fit_transform(X[:,5])

onehotencoder_1 = OneHotEncoder(categorical_features=[5])
X = onehotencoder.fit_transform(X).toarray()
loan['Dependents_1'] = pd.Categorical.from_array(loan.Dependents).codes

X
'''
#Spliting data in to training and testing dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train , y_test = train_test_split(X,y,test_size = 0.25,random_state = 0)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Fitting Kernel SVM to training set
from sklearn.svm import SVC
classifier_Svm = SVC(kernel='linear',random_state=0)
classifier_Svm.fit(X_train,y_train)


#Predicting the test set result
y_pred_svm = classifier_Svm.predict(X_test)



#Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm_svm = confusion_matrix(y_test,y_pred_svm)

#Acuuracy
import numpy as np
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred_svm)*100

#Creating Kfold for Kernel SVM regression
from sklearn.model_selection import cross_val_score
accuracy_svm = cross_val_score(estimator = classifier_Svm,X =X_train,y = y_train, cv = 10 )
#check 10 different accuracy of the model
accuracy_svm

#Average of all the models
accuracy_svm.mean()

#Standard Deviation
accuracy_svm.std()

