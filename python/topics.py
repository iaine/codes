
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel, LsiModel
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from nltk.corpus import stopwords

from wordcloud import WordCloud, STOPWORDS

import pyLDAvis
import pyLDAvis.gensim

import pandas as pd

files = pd.read_csv("./data/noracism.csv")
files.columns = ['files', 'id', 'sentence']

common_texts = []

for document in files["sentence"].to_list():

    results = []
    for token in simple_preprocess(remove_stopwords(document)):
        results.append(token)
    common_texts.append(results)

#create a common dictionary with them. 
dictionary = Dictionary()
corpus = [dictionary.doc2bow(text, allow_update=True) for text in common_texts]

lda = LdaModel(corpus, num_topics=4, id2word=dictionary, iterations=5)

topics = lda.show_topics(formatted=False)

for k,v in topics:
    print("{} -> {}".format(k, " ".join("{} : {}".format(i[0], i[1]) for i in v)))


cols = [color for name, color in mcolors.XKCD_COLORS.items()] 

cloud = WordCloud(stopwords=stopwords,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.savefig('wordcloud.png')

texts_vis_data = pyLDAvis.gensim.prepare(lda, corpus, dictionary)

pyLDAvis.display(texts_vis_data)