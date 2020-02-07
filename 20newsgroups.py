from sklearn.datasets import fetch_20newsgroups
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as skd
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from scipy.sparse import csr_matrix
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import metrics
from sklearn.metrics import accuracy_score
import ssl
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#Estrazione documenti

ssl._create_default_https_context = ssl._create_unverified_context

news_train = fetch_20newsgroups(subset='train', shuffle=True)
news_test = fetch_20newsgroups(subset='test', shuffle=True)

#Tokenizzazione
count_vactorizer = CountVectorizer()
X_train = count_vactorizer.fit_transform(news_train.data)
print("documenti: %d, parole: %d" % X_train.shape)
X_test = count_vactorizer.transform(news_test.data)
print("documenti: %d, parole: %d" % X_test.shape)

#Calcolo indice tfidf

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train)
X_test_tfidf = tfidf_transformer.fit_transform(X_test)
print("%d documenti" % len(news_test.filenames))
print("%d categorie" % len(news_test.target_names))

#Predizione e analisi dei risultati

multinomial = []
bernoulli = []

iter = [1, 100, 500, 1000, 2000, 3000, 5000, 10000, 20000, 50000, 80000, 100000]

for k in iter:
    selector = SelectKBest(chi2, k=k)
    X_new_train = selector.fit_transform(X_train_tfidf,news_train.target)
    print(X_new_train.shape)
    X_new_test = selector.transform(X_test_tfidf)

    clf = MultinomialNB()
    clf.fit(X_new_train, news_train.target)
    predicted = clf.predict(X_new_test)

    clfb = BernoulliNB()
    clfb.fit(X_new_train, news_train.target)
    predictedb = clfb.predict(X_new_test)

    multinomial.append(accuracy_score(news_test.target, predicted))
    print("Accuracy:", accuracy_score(news_test.target, predicted))
    print(metrics.classification_report(news_test.target, predicted, target_names=news_test.target_names, zero_division=True))

    bernoulli.append(accuracy_score(news_test.target, predictedb))
    print("Accuracy:", accuracy_score(news_test.target, predictedb))
    print(metrics.classification_report(news_test.target, predictedb, target_names=news_test.target_names, zero_division=True))

plt.plot(iter, multinomial, label = 'multinomial')
plt.plot(iter, bernoulli, label = 'bernoulli')
plt.legend()
plt.show()

index = iter
data = np.array([index, multinomial, bernoulli])
data = data.T
np.savetxt("20newsgroups.csv", data, fmt="%1.0f %1.5f %1.5f")