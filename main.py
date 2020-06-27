import numpy as np
import sys
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

def solve():
    training = []
    annotation = []

    with open("trainingdata.txt") as f:
        f.readline()
        for line in f:
            data = line.split()
            annotation.append(int(data[0]))
            training.append(' '.join(data[1:]))

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(training)
    tf_transformer = TfidfTransformer().fit(X_train_counts)
    tfid_transformer =TfidfTransformer()
    X_train_tfidf = tfid_transformer.fit_transform(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    clf = MultinomialNB().fit(X_train_tfidf, annotation)
    docs_new = []
    for i in range(int(input())):
        docs_new.append(input())
    X_new_count = count_vect.transform(docs_new)
    X_new_tfidf = tfid_transformer.transform(X_new_count)
    predicted = clf.predict(X_new_tfidf)
    for pred in predicted:
        print(pred)
# print(np.mean(predicted == annotation))
if __name__ == '__main__':
    solve()
