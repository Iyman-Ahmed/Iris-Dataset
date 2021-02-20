import pandas as pd
import numpy as np
dataset = pd.read_csv('Iris.csv')
X = dataset.iloc[:,1:-1].values
Y =dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print(X_train,X_test)

#from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
#classifier = RandomForestClassifier(n_estimators=10)
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train,y_train)
pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix
print(np.concatenate((pred.reshape(len(pred),1),y_test.reshape(len(y_test),1)),1))
print(accuracy_score(pred,y_test))