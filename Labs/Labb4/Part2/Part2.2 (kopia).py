from sklearn.datasets import load_files
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import nltk
import sklearn
import numpy as np
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups

# ------------ Load movie_reviews corpus data through sklearn ---------

moviedir = r'/Users/afra/Desktop/Dokument/TNM108 - Maskininlärning/Labbar/Labb4/Part2/movie_reviews'

# loading all files.
movie = load_files(moviedir, shuffle=True)

print('len(movie.data)', len(movie.data))

print('movie.target_names', movie.target_names)


# Split data into training and test sets
movie_data_train, movie_data_test, movie_target_train, movie_target_test = train_test_split(movie.data, movie.target,
                                                                                            test_size=0.20, random_state=12)
# Make SVM classifier with pipeline
clf_pipe = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42, max_iter=5, tol=None)),
])
#print(clf_pipe)

clf_pipe.fit(movie_data_train, movie_target_train)

predicted = clf_pipe.predict(movie_data_test)
print("SVM accuracy ", np.mean(predicted == movie_target_test))

print(metrics.classification_report(movie_target_test,
                                    predicted, target_names=movie.target_names))

print(metrics.confusion_matrix(movie_target_test, predicted))

# Make wow (parameters)
params = {
    'vect__ngram_range': [(1, 1), (1, 2)],
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1e-2, 1e-3)
}

# Make nice (grid search)
grid_search_clf = GridSearchCV(clf_pipe, params, cv=5, n_jobs=-1)

gs_clf = grid_search_clf.fit(movie_data_train[:400], movie_target_train[:400])

# Fake movie reviews
reviews_new = ['This movie was excellent', 'Absolute joy ride',
               'Steven Seagal was terrible', 'Steven Seagal shone through.',
               'This was certainly a movie', 'Two thumbs up', 'I fell asleep halfway through',
               "We can't wait for the sequel!!", '!', '?', 'I cannot recommend this highly enough',
               'instant classic.', 'Steven Seagal was amazing. His performance was Oscar-worthy.']


# print('\n')
#print(movie.target_names[grid_search_clf.predict(reviews_new)[0]])

print('\ngrid_search_clf.best_score_:', gs_clf.best_score_, '\n')

for param_name in sorted(params.keys()):
    print("%s: %r" % (param_name, grid_search_clf.best_params_[param_name]))

pred = gs_clf.predict(reviews_new)

# Print results
print('\n')
for review, category in zip(reviews_new, pred):
    print('%r => %s' % (review, movie.target_names[category]))