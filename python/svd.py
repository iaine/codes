from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import pandas as pd

files = pd.read_csv("./data/noracism.csv")
files.columns = ['files', 'id', 'sentence']
docs = files["sentence"].to_list()

num_clusters=8

#https://stackoverflow.com/questions/28160335/plot-a-document-tfidf-2d-graph/28205420#28205420

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
])        
X = pipeline.fit_transform(docs)

pca = TruncatedSVD(n_components=num_clusters).fit(X)
data2D = pca.transform(X)

plt.scatter(data2D[:,0], data2D[:,1], cmap='gray') 

kmeans = KMeans(n_clusters=num_clusters).fit(X)
centers2D = pca.transform(kmeans.cluster_centers_)

plt.scatter(centers2D[:,0], centers2D[:,1], 
            marker='x', s=200, linewidths=3, c='r', cmap='gray')

labels = range(num_clusters)
for label, x, y in zip(labels, centers2D[:,0], centers2D[:,1]):
    plt.text(x * (1 + 0.01), y * (1 + 0.01) , label, fontsize=12)


plt.savefig('svd.png')