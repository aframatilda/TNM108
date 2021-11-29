import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import MinMaxScaler 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage 

customer_data = pd.read_csv('shopping_data.csv')

print(customer_data.shape)
print(customer_data.head())

data = customer_data.iloc[:, 3:5].values
print(data.shape)
print(data)

# labels = range(1, 201)
# plt.figure(figsize=(10, 7)) 
# plt.subplots_adjust(bottom=0.1) 
# plt.scatter(data[:,0],data[:,1], label='True Position') 
# for label, x, y in zip(labels, data[:, 0], data[:, 1]):
#     plt.annotate(label,xy=(x, y),xytext=(-3, 3),textcoords='offset points', ha='right',va='bottom') 
# plt.show()

linked = linkage(data, 'ward')
labelList = range(1, 201)
plt.figure(figsize=(10, 7)) 
dendrogram(linked, orientation='top',
labels=labelList, distance_sort='descending', show_leaf_counts=True)
plt.show()

cluster = AgglomerativeClustering(n_clusters=5, 
    affinity='euclidean', 
    linkage='ward') 
cluster.fit_predict(data)

plt.scatter(data[:,0],data[:,1], c=cluster.labels_, cmap='rainbow') 
plt.show()

