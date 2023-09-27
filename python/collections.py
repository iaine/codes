from collections import Counter
from nltk.corpus import stopwords

import pandas as pd

stops = set(stopwords.words('english'))
stops.update(set(["No", "etc"]))

files = pd.read_csv("./data/noracism.csv")
files.columns = ['files', 'id', 'sentence']

words = Counter()

for document in files["sentence"].to_list():
    doc = document.replace(',',' ').replace('.',' ').replace('\r\n',' ').split(' ')
    word = [wd for wd in doc if wd not in stops and wd != ""]
    words.update(word)

print(words.most_common(15))
fh = open("common.csv", "w+")
for key, count in words.items():
    fh.write("{},{}\n".format(key, count))
fh.close()