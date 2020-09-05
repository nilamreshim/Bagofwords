import os
from sklearn.feature_extraction.text import CountVectorizer
import scipy
from sklearn.feature_extraction import text

# Step 1 :- Get all the files from the folder
myPath = os.getcwd() + "\\Articles"

for file in os.listdir(myPath):
    f = open(myPath + "\\" + file, "r")
    document = []
    temp=""

    # step 2 :- read the content in to document collection
    for line in f.readlines():
        temp = temp + line

    document.append(temp)

    # step 3 :- pass this document to vectorizer
    vectorizer = CountVectorizer(stop_words=text.ENGLISH_STOP_WORDS)
    # countvectorizer will take the document
    # content and train
    counts = vectorizer.fit_transform(document)
    bows = vectorizer.vocabulary_
    coo = scipy.sparse.coo_matrix(counts)

    fbow = open( os.path.splitext(f.name)[0] + "bow.txt", "w")

    for _count, _bowname in zip(coo.data, bows.keys()):
        if (_count > 2):
            fbow.write(str(_count) + "  -- " + _bowname + "\n")

    fbow.close()