from nltk.corpus import reuters
from sklearn.multiclass import OneVsRestClassifier
import nltk
import ssl
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('reuters')
nltk.download('punkt')

#funzione di tokenizzazione

def tokenize(text):
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text))
    tokens = (list(map(lambda token: PorterStemmer().stem(token), words)))
    p = re.compile('[a-zA-Z]+')
    filtered_tokens = list(filter(lambda token: p.match(token) and len(token) >= min_length, tokens))
    return filtered_tokens

#Lista di Documenti

categories = ['acq', 'corn', 'crude', 'earn', 'grain', 'interest', 'money-fx', 'ship', 'trade', 'wheat']

earn = reuters.fileids('earn')
acq = reuters.fileids('acq')
moneyfx = reuters.fileids('money-fx')
grain = reuters.fileids('grain')
crude = reuters.fileids('crude')
trade = reuters.fileids('trade')
interest = reuters.fileids('interest')
wheat = reuters.fileids('wheat')
ship = reuters.fileids('ship')
corn = reuters.fileids('corn')
documents = []

for i in range(0, len(earn)):
    documents.append(earn[i])
for i in range(0, len(acq)):
    documents.append(acq[i])
for i in range(0, len(moneyfx)):
    documents.append(moneyfx[i])
for i in range(0, len(grain)):
    documents.append(grain[i])
for i in range(0, len(crude)):
    documents.append(crude[i])
for i in range(0, len(trade)):
    documents.append(trade[i])
for i in range(0, len(interest)):
    documents.append(interest[i])
for i in range(0, len(wheat)):
    documents.append(wheat[i])
for i in range(0, len(ship)):
    documents.append(ship[i])
for i in range(0, len(corn)):
    documents.append(corn[i])

print(str(len(documents)) + " documenti")

#Divisione tra Train e Test

train_docs_id = [d for d in documents if d.startswith('training/')]
print(str(len(train_docs_id)) + " documenti di train")

test_docs_id = [d for d in documents if d.startswith('test/')]
print(str(len(test_docs_id)) + " documenti di test")

train_docs = [reuters.raw(doc_id) for doc_id in train_docs_id]
test_docs = [reuters.raw(doc_id) for doc_id in test_docs_id]

#Tokenizzazione

vectorizer = TfidfVectorizer(tokenizer = tokenize)
vectorised_train_documents = vectorizer.fit_transform(train_docs)
vectorised_test_documents = vectorizer.transform(test_docs)
print(vectorised_train_documents.shape)
print(vectorised_test_documents.shape)

#Calcolo indice tfifd

tfidf_transformer = TfidfTransformer()
vectorised_train_tfidf_documents = tfidf_transformer.fit_transform(vectorised_train_documents)
print(vectorised_train_tfidf_documents.shape)

#Estrazione valori attesi

mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform([reuters.categories(doc_id) for doc_id in train_docs_id])
test_labels = mlb.transform([reuters.categories(doc_id) for doc_id in test_docs_id])

#Predizione e analisi risultati

multinomial = []
bernoulli = []
macqa = []
bacqa = []
macc = []
bacc = []

iter = [170, 200, 500, 800, 1000, 2000, 5000, 10000, 15000, 18000]

for k in iter:
    selector = SelectKBest(chi2, k=k)
    X_new_train = selector.fit_transform(vectorised_train_documents,train_labels)
    print(X_new_train.shape)
    X_new_test = selector.transform(vectorised_test_documents)

    clf = OneVsRestClassifier(MultinomialNB())
    clf.fit(X_new_train, train_labels)
    predicted = clf.predict(X_new_test)

    clfb = OneVsRestClassifier(BernoulliNB())
    clfb.fit(X_new_train, train_labels)
    predictedb = clfb.predict(X_new_test)


    multinomial.append(accuracy_score(test_labels, predicted))
    print("Accuracy:", accuracy_score(test_labels, predicted))
    cm = confusion_matrix(test_labels.argmax(axis=1), predicted.argmax(axis=1))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    accuracies = cm.diagonal()
    macqa.append(accuracies[0])

    bernoulli.append(accuracy_score(test_labels, predictedb))
    print("Accuracy:", accuracy_score(test_labels, predictedb))
    cmb = confusion_matrix(test_labels.argmax(axis=1), predictedb.argmax(axis=1))
    cmb = cmb.astype('float') / cmb.sum(axis=1)[:, np.newaxis]
    accuraciesb = cmb.diagonal()
    bacqa.append(accuraciesb[0])

plt.plot(iter, multinomial, label = 'multinomial')
plt.plot(iter, bernoulli, label = 'bernoulli')
plt.legend()
plt.show()

plt.plot(iter, macqa, label = 'multinomial')
plt.plot(iter, bacqa, label = 'bernoulli')
plt.ylabel('acq')
plt.legend()
plt.show()

index = iter
data = np.array([index, multinomial, bernoulli])
data = data.T
np.savetxt("Reuters.csv", data, fmt="%1.0f %1.5f %1.5f")

index = iter
data = np.array([index, macqa, bacqa])
data = data.T
np.savetxt("Acq.csv", data, fmt="%1.0f %1.5f %1.5f")
