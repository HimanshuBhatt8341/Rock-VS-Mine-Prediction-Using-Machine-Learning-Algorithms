import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
data=pd.read_csv('sonar.csv',header=None)
data.head()
data.tail()
data.shape
data.info()
data.isnull().sum()
data.describe()
data.columns
sns.countplot(x=data[60])
data.groupby(60).mean()
x=data.drop(60,axis=1)
y=data[60]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
lr=LogisticRegression()
lr.fit(x_train,y_train)
y_pred1=lr.predict(x_test)
accuracy_score(y_test,y_pred1)
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
y_pred2=knn.predict(x_test)
accuracy_score(y_test,y_pred2)
rf=RandomForestClassifier()
rf.fit(x_train,y_train)
y_pred3=rf.predict(x_test)
accuracy_score(y_test,y_pred3)
from sklearn.linear_model import SGDClassifier
sgd=SGDClassifier()
for i in range(len(x_train)):
    sgd.partial_fit(x_train[i:i+1],y_train[i:i+1],classes=['R','M'])
    score=sgd.score(x_test,y_test)
    print("Accuracy:",score)
    final_data=pd.DataFrame({'Model':['LR','KNN','RF','SGD'],
              'ACC':[accuracy_score(y_test,y_pred1),
                     accuracy_score(y_test,y_pred2),
                     accuracy_score(y_test,y_pred3),
                     score]})
    final_data
    knn1=KNeighborsClassifier(n_neighbors=3)
knn1.fit(x,y)
joblib.dump(knn1,'rock_mine_prediction_model')
input_data = (0.0191,0.0173,0.0291,0.0301,0.0463,0.0690,0.0576,0.1103,0.2423,0.3134,0.4786,0.5239,0.4393,0.3440,0.2869,0.3889,0.4420,0.3892,0.4088,0.5006,0.7271,0.9385,1.0000,0.9831,0.9932,0.9161,0.8237,0.6957,0.4536,0.3281,0.2522,0.3964,0.4154,0.3308,0.1445,0.1923,0.3208,0.3367,0.5683,0.5505,0.3231,0.0448,0.3131,0.3387,0.4130,0.3639,0.2069,0.0859,0.0600,0.0267,0.0125,0.0040,0.0136,0.0137,0.0172,0.0132,0.0110,0.0122,0.0114,0.0068)
input_data_np_array = np.asarray(input_data)
reshaped_input = input_data_np_array.reshape(1,-1)
prediction = knn1.predict(reshaped_input)

if prediction[0] == 'R':
    print('The Object is a Rock')
else: 
    print('The Object is a Mine')
    