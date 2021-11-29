import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('data_cars.csv',header=None)
for i in range(len(df.columns)):
   df[i] = df[i].astype('category')
#print(df.head())

#map catgories to values
map0 = dict( zip( df[0].cat.categories, range( len(df[0].cat.categories ))))
map1 = dict( zip( df[1].cat.categories, range( len(df[1].cat.categories ))))
map2 = dict( zip( df[2].cat.categories, range( len(df[2].cat.categories ))))
map3 = dict( zip( df[3].cat.categories, range( len(df[3].cat.categories ))))
map4 = dict( zip( df[4].cat.categories, range( len(df[4].cat.categories ))))
map5 = dict( zip( df[5].cat.categories, range( len(df[5].cat.categories ))))
map6 = dict( zip( df[6].cat.categories, range( len(df[6].cat.categories ))))

cat_cols = df.select_dtypes(['category']).columns
df[cat_cols] = df[cat_cols].apply(lambda x: x.cat.codes)
df = df.iloc[np.random.permutation(len(df))]
#print(df.head())

df_f1 = pd.DataFrame(columns=['method']+sorted(map6, key=map6.get))
df_precision = pd.DataFrame(columns=['method']+sorted(map6, key=map6.get))
df_recall = pd.DataFrame(columns=['method']+sorted(map6, key=map6.get))

def CalcMeasures(method,y_pred,y_true,df_f1=df_f1,df_precision=df_precision, df_recall=df_recall):
    df_f1.loc[len(df_f1)] = [method]+list(f1_score(y_true,y_pred,average=None))
    df_precision.loc[len(df_precision)] = [method]+list(precision_score(y_pred,y_true,average=None))
    df_recall.loc[len(df_recall)] = [method]+list(recall_score(y_pred, y_true,average=None))

X = df[df.columns[:-1]].values
Y = df[df.columns[-1]].values

cv = 10

method = 'linear support vector machine'
clf = svm.SVC(kernel='linear',C=50)
y_pred = cross_val_predict(clf, X,Y, cv=cv)
CalcMeasures(method,y_pred,Y)

method = 'naive bayes'
clf = MultinomialNB()
y_pred = cross_val_predict(clf, X,Y, cv=cv)
CalcMeasures(method,y_pred,Y)

method = 'logistic regression'
clf = LogisticRegression()
y_pred = cross_val_predict(clf, X,Y, cv=cv)
CalcMeasures(method,y_pred,Y)

method = 'k nearest neighbours'
clf = KNeighborsClassifier(weights='distance',n_neighbors=5)
y_pred = cross_val_predict(clf, X,Y, cv=cv)
CalcMeasures(method,y_pred,Y)

method = 'Decision tree'
clf = DecisionTreeClassifier(random_state = 0)
y_pred = cross_val_predict(clf, X,Y, cv=cv)
CalcMeasures(method,y_pred,Y)

# radial basis function svm
method = 'RBF SVM'
clf = svm.SVC(kernel='rbf', C=50, gamma="auto")
y_pred = cross_val_predict(clf, X,Y, cv=cv)
CalcMeasures(method,y_pred,Y)

method = 'random forest'
clf = RandomForestClassifier(random_state=0)
y_pred = cross_val_predict(clf, X,Y, cv=cv)
CalcMeasures(method,y_pred,Y)

print(df_f1) 
print('\n')
print(df_precision)
print('\n')
print(df_recall)
print('\n')


labels_counts=df[6].value_counts()
print(pd.Series(map6).map(labels_counts))