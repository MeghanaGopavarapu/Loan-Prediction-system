import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import missingno as msno
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import pickle

df=pd.read_csv('D:\\train.csv' )
msno.matrix(df)

df.head()
sns.boxplot(df['LoanAmount'])
plt.show()

del df['LoanAmount']
df.info()
df.dtypes
categorical=['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
numerical=['ApplicantIncome','CoapplicantIncome','Loan_Amount_Term','Credit_History']


for i in range(len(categorical)):
    chi_square=stats.chi2_contingency(pd.crosstab(df[categorical[i]],df.Loan_Status))
    res=stats.chi2.pdf(3.84,chi_square[2])
    print(df[categorical[i]],'chi-square value:',res)
    if res>0.05:# if the value is greater than 0.05 then class label and variable is closely related so delete the col
        del df[categorical[i]]
df.info()
del df['Loan_ID']


#df["ApplicantIncome"].hist()
#filling missing values with mode
missing=['Gender','Married','Education','Self_Employed','Loan_Amount_Term','Credit_History']
for i in missing:
    df[i]=df[i].fillna(df[i].dropna().mode()[0])
df.info()


#df['Loan_ID']=df['Loan_ID'].str.lstrip('LP00')
#changing categorical variables into numerical variables
#df['Loan_ID'] = df['Loan_ID'].astype(np.int)
df['Gender'] = df['Gender'].map({'Female':0,'Male':1})
df['Married'] = df['Married'].map({'No':0,'Yes':1})
df['Education'] = df['Education'].map({'Not Graduate':0,'Graduate':1})
df['Self_Employed'] = df['Self_Employed'].map({'No':0,'Yes':1})
df['Loan_Status'] = df['Loan_Status'].map({'N':0,'Y':1})
df.dtypes
df.shape
df.info()


#heat map generation
plt.figure(figsize=(10,5))
sns.heatmap(df.corr(),annot=True)
del df['ApplicantIncome']
df.info()

#standard scaler
df[['CoapplicantIncome','Loan_Amount_Term','Credit_History']] = StandardScaler().fit_transform(df[['CoapplicantIncome','Loan_Amount_Term','Credit_History']])
df

#splitting dataset into training and test data
X=df.iloc[:,1:8].values
Y=df.iloc[:,-1].values
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

X_train.shape
X_test.shape
pca = PCA(n_components = 2)   
x_train = pca.fit_transform(X_train) 
x_test = pca.transform(X_test) 


# logistic regression object 
lr = LogisticRegression() 
# train the model on train set 
lr.fit(x_train, y_train.ravel()) 

predictions = lr.predict(x_test) 


# print classification report 
print(classification_report(y_test, predictions)) 
print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1))) 
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0))) 


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 2) 
x_train_res, y_train_res = sm.fit_resample(x_train, y_train.ravel()) 


print('After OverSampling, the shape of train_X: {}'.format(x_train_res.shape)) 
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape)) 
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1))) 
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0))) 


lr1 = LogisticRegression() 
lr1.fit(x_train_res, y_train_res.ravel()) 


predictions = lr1.predict(x_test) 
sns.countplot(df.Loan_Status)
plt.show()


# print classification report 
print(classification_report(y_test, predictions)) 

classifier = LogisticRegression(random_state = 0)
trained_model=classifier.fit(x_train,y_train)
trained_model.fit(x_train,y_train )


# Predicting the Test set results
y_pred = classifier.predict(x_test)

# Making the Confusion Matrix
cm1 = confusion_matrix(y_test, y_pred)
print(cm1)

#print("Accuracy score of train LogisticRegression")
#print(accuracy_score(y_train, trained_model.predict(x_train))*100)
print("Accuracy score of test LogisticRegression")
a1=accuracy_score(y_test, y_pred)*100
print(a1)


#decision tree algorithm

classifier = DecisionTreeClassifier(random_state = 0)
trained_model=classifier.fit(x_train,y_train)
trained_model.fit(x_train,y_train )
y_pred = classifier.predict(x_test)
cm2= confusion_matrix(y_test, y_pred)
print(cm2)


print("Accuracy score of test Decision tree")
a2=accuracy_score(y_test, y_pred)*100
print(a2)

#RandomForestClassifier algorithm

classifier = RandomForestClassifier(random_state = 0)
trained_model=classifier.fit(x_train,y_train)
trained_model.fit(x_train,y_train )
y_pred = classifier.predict(x_test)
cm3 = confusion_matrix(y_test, y_pred)
print(cm3)

print("Accuracy score of test RandomForestClassifier")
a3=accuracy_score(y_test, y_pred)*100
print(a3)


#support vector machine algorithm

classifier = SVC()
trained_model=classifier.fit(x_train,y_train)
trained_model.fit(x_train,y_train )
y_pred = classifier.predict(x_test)
cm5 = confusion_matrix(y_test, y_pred)
print(cm5)

print("Accuracy score of test support vector machine(svm)")
a5=accuracy_score(y_test, y_pred)*100
print(a5)


#naive bayes algorithm

classifier = GaussianNB()
trained_model=classifier.fit(x_train,y_train)
trained_model.fit(x_train,y_train )
y_pred = classifier.predict(x_test)
cm6 = confusion_matrix(y_test, y_pred)
print(cm6)

print("Accuracy score of test Naive bayes")
a6=accuracy_score(y_test, y_pred)*100
print(a6)

classifier = KNeighborsClassifier()
trained_model=classifier.fit(x_train,y_train)
trained_model.fit(x_train,y_train )
# Predicting the Test set results
y_pred = classifier.predict(x_test)
# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("Accuracy score of test KNeighborsClassifier")
acc=accuracy_score(y_test, y_pred)*100
print(acc)

# x-coordinates of left sides of bars 
left = [0,20,40,60,80,100] 
# heights of bars 
height = [a1,a2,a3,acc,a5,a6] 
# labels for bars 
tick_label = ['LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier', 'KNeighborsClassifier', 'SupportVectorMachine','NaiveBayes'] 
plt.figure(figsize=(15, 8))
# plotting a bar chart 
plt.bar(left, height, tick_label = tick_label, 
		width = 15, color = ['lightpink','lightblue','yellow','orange','lightgreen','purple']) 
# naming the x-axis 
plt.xlabel('algorithms') 
# naming the y-axis 
plt.ylabel('accuracy') 
# plot title 
plt.title('comparision of algorithms') 
# function to show the plot 
plt.show() 


classifier = RandomForestClassifier()
#Fitting model with trainig data
classifier.fit(X_train,y_train)
# Saving model to disk
pickle.dump(classifier, open('model.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[2,3,4,5,6,7,8]]))