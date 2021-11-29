from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import sklearn
from sklearn.datasets import load_files

# ------------ Load movie_reviews corpus data through sklearn ---------

moviedir = r'/Users/afra/Desktop/Dokument/TNM108 - Maskininlärning/Labbar/Labb4/Part2/movie_reviews'

# loading all files.
movie = load_files(moviedir, shuffle=True)
# print(len(movie.data))

# target names ("classes") are automatically generated from subfolder names
# print(movie.target_names)

# First file seems to be about a Schwarzenegger movie.
# print(movie.data[0][:500])

# first file is in "neg" folder
# print(movie.filenames[0])

# first file is a negative review and is mapped to 0 index 'neg' in target_names
# print(movie.target[0])


# ------------- A detour: try out CountVectorizer & TF-IDF -------------

# Turn off pretty printing of jupyter notebook... it generates long lines
# %pprint

# Three tiny "documents"
docs = ['A rose is a rose is a rose is a rose.',
        'Oh, what a fine day it is.',
        "A day ain't over till it's truly over."]

# Initialize a CountVectorizer to use NLTK's tokenizer instead of its
#    default one (which ignores punctuation and stopwords).
# Minimum document frequency set to 1.
fooVzer = CountVectorizer(min_df=1, tokenizer=nltk.word_tokenize)

# .fit_transform does two things:
# (1) fit: adapts fooVzer to the supplied text data (rounds up top words into vector space)
# (2) transform: creates and returns a count-vectorized output of docs
docs_counts = fooVzer.fit_transform(docs)
# print(docs_counts)

# fooVzer now contains vocab dictionary which maps unique words to indexes
fooVzer.vocabulary_
# print(fooVzer.vocabulary_)

# docs_counts has a dimension of 3 (document count) by 16 (# of unique words)
docs_counts.shape
# print(docs_counts.shape)

# this vector is small enough to view in a full, non-sparse form!
docs_counts.toarray()
# print(docs_counts.toarray())

# Convert raw frequency counts into TF-IDF (Term Frequency -- Inverse Document Frequency) values
fooTfmer = TfidfTransformer()

# Again, fit and transform
docs_tfidf = fooTfmer.fit_transform(docs_counts)

# TF-IDF values
# raw counts have been normalized against document length,
# terms that are found across many docs are weighted down ('a' vs. 'rose')
docs_tfidf.toarray()
# print(docs_tfidf.toarray())


# --------- Back to real data: movie reviews ----------

# Split data into training and test sets
docs_train, docs_test, y_train, y_test = train_test_split(movie.data, movie.target,
                                                          test_size=0.20, random_state=12)
# initialize CountVectorizer
# use top 3000 words only. 78.25% acc.
movieVzer = CountVectorizer(
    min_df=2, tokenizer=nltk.word_tokenize, max_features=3000)
# movieVzer = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize)         # use all 25K words. Higher accuracy

# fit and tranform using training text
docs_train_counts = movieVzer.fit_transform(docs_train)
# 'screen' is found in the corpus, mapped to index 2290
t = movieVzer.vocabulary_.get('screen')
# print(t)

# Likewise, Mr. Steven Seagal is present...
mv = movieVzer.vocabulary_.get('seagal')
# print(mv)

# Convert raw frequency counts into TF-IDF values
movieTfmer = TfidfTransformer()
docs_train_tfidf = movieTfmer.fit_transform(docs_train_counts)

# huge dimensions! 1,600 documents, 3K unique terms.
d = docs_train_counts.shape
# print(d)

# Using the fitted vectorizer and transformer, tranform the test data
docs_test_counts = movieVzer.transform(docs_test)
docs_test_tfidf = movieTfmer.transform(docs_test_counts)


# ----------- Training and testing a Naive Bayes classifier ---------

# Now ready to build a classifier.
# We will use Multinominal Naive Bayes as our model
# Train a Multimoda Naive Bayes classifier. Again, we call it "fitting"
clf = MultinomialNB()
clf.fit(docs_train_tfidf, y_train)

# Predict the Test set results, find accuracy
y_pred = clf.predict(docs_test_tfidf)
sklearn.metrics.accuracy_score(y_test, y_pred)
# print(sklearn.metrics.accuracy_score(y_test, y_pred))

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
# print(cm)

# ------------ Trying the classifier on fake movie reviews -----------

# very short and fake movie reviews
reviews_new = ['This movie was excellent', 'Absolute joy ride',
               'Steven Seagal was terrible', 'Steven Seagal shone through.',
               'This was certainly a movie', 'Two thumbs up', 'I fell asleep halfway through',
               "We can't wait for the sequel!!", '!', '?', 'I cannot recommend this highly enough',
               'instant classic.', 'Steven Seagal was amazing. His performance was Oscar-worthy.']

# turn text into count vector
reviews_new_counts = movieVzer.transform(reviews_new)

# turn into tfidf vector
# have classifier make a prediction
reviews_new_tfidf = movieTfmer.transform(reviews_new_counts)

pred = clf.predict(reviews_new_tfidf)

# print out results
for review, category in zip(reviews_new, pred):
    print('%r => %s' % (review, movie.target_names[category]))

# Mr. Seagal simply cannot win!