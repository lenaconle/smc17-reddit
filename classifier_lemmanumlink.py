# Dieser Classifier benutzt die Politik-Kommentare mit einer balancierten Verteilung und lemmatisiert, ersetzt die URL und entfernt Zahlen vor dem Training

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from random import shuffle
from nltk.stem.wordnet import WordNetLemmatizer

# Datenextraktion und Vorverarbeitung:
lmtzr = WordNetLemmatizer()
commentlist = []
with open('final_dataset.json') as data_file:
    for line in data_file:
        commentjson = json.loads(line)
        commentlist.append(commentjson)
shuffle(commentlist)
dataset = []
for comment in commentlist:
    relinfo = []
    relinfo.append(comment['body'])
    if comment['controversiality'] == 1:
        relinfo.append("controverse")
    else:
        relinfo.append("non-controverse")
    dataset.append(relinfo)
shuffle(dataset)

# Entfernung Zahlen:
for comment in dataset:
    s = comment[0]
    comment[0] = "".join([i for i in s if not i.isdigit()])

# Lemmatisierung:
for comment in dataset:
    comment_words = comment[0].split()
    comment_words = [w.replace(w,lmtzr.lemmatize(w)) for w in comment_words]
    comment[0] = " ".join(i for i in comment_words)

# URL ersetzen:
linkkeywords = ['www.','http','.com']
for comment in dataset:
    comment_words = comment[0].split()
    for word in comment_words:
        if any(x in word for x in linkkeywords):
            comment_words = [w.replace(word,"lliinnkk") for w in comment_words]
    comment[0] = " ".join(i for i in comment_words)

texts_train = []
labels_train = []
texts_test = []
labels_test = []

# Verteilung der Daten auf die Trainings- und Testdaten:
for i in range(76400):
    texts_train.append(dataset[i][0])
    labels_train.append(dataset[i][1])
for i in range(76400,len(dataset)):
    texts_test.append(dataset[i][0])
    labels_test.append(dataset[i][1])

# Vectorizer und Training:
vect = TfidfVectorizer(ngram_range=(1, 3))
X_train = vect.fit_transform(texts_train)
y_train = labels_train
clf = LinearSVC()
clf.fit(X_train,y_train)

# Prediction:
X_test = vect.transform(texts_test)
predictions = clf.predict(X_test)
print "Overall Accuracy:"
print accuracy_score(labels_test,predictions)

# Funktion zum Ausgeben der most informative features:
def show_most_informative_features(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2) 

# Ausgeben des Classification Reports:
print "Classification Report:"
print(classification_report(labels_test, predictions))

# Ausgeben der most informative features:
show_most_informative_features(vect, clf)
