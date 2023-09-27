'''
Function to create a dendrogram from a set of sentences. 
'''
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import matplotlib.pyplot as plt

files = pd.read_csv("./data/noracism.csv")
files.columns = ['files', 'id', 'sentence']

#ver y,uch like early Stanford Lit aLab
cv = CountVectorizer(stop_words='english')
cv_matrix = cv.fit_transform(files["sentence"])
#listoffiles = glob("codes/*.txt")
#cv_matrix = cv.fit_transform(listoffiles, input='filename')

df_dtm = pd.DataFrame(cv_matrix.toarray(), index = files["files"], columns = cv.get_feature_names_out())

tree = linkage(df_dtm, method='complete', metric='euclidean')

plt.figure(figsize=(15,17.5))
plt.title = "Dendrogram following no racism phrase in Mastodon codes of conduct written in English"
dendrogram(tree, leaf_rotation=270, labels=files["files"].tolist())
plt.savefig('no_racism_phrase.png', format='png', bbox_inches='tight')