import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
###############################
data=pd.read_csv('insurance.csv')
###############################
data.shape
data.head()
data.tail()
data.columns
data.info()
data.isna().sum()
data.describe()
###############################
sns.heatmap(data.loc[:,['age','bmi','children','charges']].corr(),annot=True)
###############################
plt.scatter(x=['age'],y=['charges'],color='Yellow')
plt.scatter(x=['bmi'],y=['charges'],color='Blue')
plt.scatter(x=['children'],y=['charges'],color='Black')
###############################
data['sex'].unique()
data['sex'].value_counts()
###############################
data['smoker'].unique()
data['smoker'].value_counts()
###############################
data['region'].unique()
data['region'].value_counts()
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
data['sex']=encoder.fit_transform(data['sex'])
data['smoker']=encoder.fit_transform(data['smoker'])
data['region']=encoder.fit_transform(data['region'])
###############################
x=data.drop(['charges'],axis=1)
y=data['charges']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               test_size=0.20,random_state=0)
###############################
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
regressor.coef_
regressor.intercept_
##############################
plt.plot(x_train,y_train,color='Blue')
plt.plot(x_train,regressor.predict(x_train),color='Yellow')
plt.title('Data vs charges')
plt.xlabel('Charges')
plt.ylabel('Data')

##############################
plt.plot(x_test,y_test,color='Green')
plt.plot(x_train,regressor.predict(x_train),color='Violet')
plt.title('Data vs charges')
plt.xlabel('Charges')
plt.ylabel('Data')