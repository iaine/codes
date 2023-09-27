from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import pandas as pd

files = pd.read_csv("./data/noracism.csv")
files.columns = ['files', 'id', 'sentence']
docs = files["sentence"].to_list()

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(docs)

true_k = 8
random_state = 1
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=10, random_state=random_state)
model.fit(X)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()
for i in range(true_k):
    print ("Cluster %d:" % i,)
    for ind in order_centroids[i, :10]:
        print (' %s' % terms[ind],)

#https://stackoverflow.com/questions/43541187/how-can-i-plot-a-kmeans-text-clustering-result-with-matplotlib
centroids = model.cluster_centers_
tsne_init = 'random'  # could also be 'random'
tsne_perplexity = 2.0
tsne_early_exaggeration = 4.0
tsne_learning_rate = 1000
model = TSNE(n_components=2, random_state=random_state, init=tsne_init, perplexity=tsne_perplexity,
         early_exaggeration=tsne_early_exaggeration, learning_rate=tsne_learning_rate)

transformed_centroids = model.fit_transform(centroids)
plt.scatter(transformed_centroids[:, 0], transformed_centroids[:, 1], marker='x')
plt.savefig('tsne.png')