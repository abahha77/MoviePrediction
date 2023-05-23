import pickle
import time
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import f_classif
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import ast
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MinMaxScaler


movies = pd.read_csv('C:/Users/Ahmed Baha/PycharmProjects/MachineProjectClassification/movies-classification/movies-classification/movies-classification-dataset.csv')
credit = pd.read_csv('C:/Users/Ahmed Baha/PycharmProjects/MachineProjectClassification/movies-classification/movies-classification/movies-credit-students-train.csv')


credit.columns = ['id', 'title', 'cast', 'crew']
data = movies.merge(credit, on='id')
data = data.loc[:, ~data.T.duplicated(keep='first')]
X = data.drop(['Rate'], axis=1)
y = data['Rate']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


label_encoder = LabelEncoder()
label_encoder.fit(y_train)


y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)

X_train['genres'] = X_train['genres'].map(lambda x: ast.literal_eval(x))
X_train['genres'] = X_train['genres'].map(lambda x: ' '.join([i.get('name') for i in x]) if len(x) > 0 else np.NaN)
X_test['genres'] = X_test['genres'].map(lambda x: ast.literal_eval(x))
X_test['genres'] = X_test['genres'].map(lambda x: ' '.join([i.get('name') for i in x]) if len(x) > 0 else np.NaN)


X_train['keywords'] = X_train['keywords'].map(lambda x: ast.literal_eval(x))
X_train['keywords'] = X_train['keywords'].map(lambda x: ' '.join([i.get('name') for i in x]) if len(x) > 0 else np.NaN)
X_test['keywords'] = X_test['keywords'].map(lambda x: ast.literal_eval(x))
X_test['keywords'] = X_test['keywords'].map(lambda x: ' '.join([i.get('name') for i in x]) if len(x) > 0 else np.NaN)


X_train['production_countries'] = X_train['production_countries'].map(lambda x: ast.literal_eval(x))
X_train['production_countries'] = X_train['production_countries'].map(lambda x: ' '.join([i.get('iso_3166_1') for i in x]) if len(x) > 0 else np.NaN)
X_test['production_countries'] = X_test['production_countries'].map(lambda x: ast.literal_eval(x))
X_test['production_countries'] = X_test['production_countries'].map(lambda x: ' '.join([i.get('iso_3166_1') for i in x]) if len(x) > 0 else np.NaN)

X_train['spoken_languages'] = X_train['spoken_languages'].map(lambda x: ast.literal_eval(x))
X_train['spoken_languages'] = X_train['spoken_languages'].map(lambda x: ' '.join([i.get('iso_639_1') for i in x]) if len(x) > 0 else np.NaN)
X_test['spoken_languages'] = X_test['spoken_languages'].map(lambda x: ast.literal_eval(x))
X_test['spoken_languages'] = X_test['spoken_languages'].map(lambda x: ' '.join([i.get('iso_639_1') for i in x]) if len(x) > 0 else np.NaN)

X_train['crew'] = X_train['crew'].map(lambda x: ast.literal_eval(x))
X_train['crew'] = X_train['crew'].map(lambda x: ' '.join([i.get('name','department') for i in x]) if len(x) > 0 else np.NaN)
X_test['crew'] = X_test['crew'].map(lambda x: ast.literal_eval(x))
X_test['crew'] = X_test['crew'].map(lambda x: ' '.join([i.get('name','department') for i in x]) if len(x) > 0 else np.NaN)

X_train['cast'] = X_train['cast'].map(lambda x: ast.literal_eval(x))
X_train['cast'] = X_train['cast'].map(lambda x: ' '.join([i.get('name') for i in x]) if len(x) > 0 else np.NaN)
X_test['cast'] = X_test['cast'].map(lambda x: ast.literal_eval(x))
X_test['cast'] = X_test['cast'].map(lambda x: ' '.join([i.get('name') for i in x]) if len(x) > 0 else np.NaN)

X_train['production_companies'] = X_train['production_companies'].map(lambda x: ast.literal_eval(x))
X_train['production_companies'] = X_train['production_companies'].map(lambda x: ' '.join([i.get('name') for i in x]) if len(x) > 0 else np.NaN)
X_test['production_companies'] = X_test['production_companies'].map(lambda x: ast.literal_eval(x))
X_test['production_companies'] = X_test['production_companies'].map(lambda x: ' '.join([i.get('name') for i in x]) if len(x) > 0 else np.NaN)


X_train.drop('homepage', axis='columns', inplace=True)
X_test.drop('homepage', axis='columns', inplace=True)

columns = ['runtime', 'genres','keywords', 'overview', 'production_companies', 'production_countries', 'spoken_languages', 'crew', 'tagline', 'original_title']

for column in columns:
    X_train[column].fillna(X_train[column].mode()[0], inplace=True)
    X_test[column].fillna(X_test[column].mode()[0], inplace=True)


X_train = X_train.loc[:, ~X_train.T.duplicated(keep='first')]
X_test = X_test.loc[:, ~X_test.T.duplicated(keep='first')]


features_to_encode = ['status', 'tagline', 'original_title', 'original_language', 'genres', 'title_x','production_countries', 'spoken_languages', 'keywords', 'cast', 'crew', 'production_companies']

label_encoder = preprocessing.LabelEncoder()

for feature in features_to_encode:
    label_encoder.fit(X_train[feature])
    X_test[feature] = X_test[feature].map(lambda s: '<unknown>' if s not in label_encoder.classes_ else s)
    label_encoder.classes_ = np.append(label_encoder.classes_, '<unknown>')
    X_train[feature] = label_encoder.transform(X_train[feature])
    X_test[feature] = label_encoder.transform(X_test[feature])

y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)


with open('label_encoding', 'wb') as f:
    pickle.dump(label_encoder, f)

X_train.head()


X_train['month'] = pd.DatetimeIndex(X_train['release_date']).month
X_train['day'] = pd.DatetimeIndex(X_train['release_date']).day
X_train['year'] = pd.DatetimeIndex(X_train['release_date']).year
X_test['month'] = pd.DatetimeIndex(X_test['release_date']).month
X_test['day'] = pd.DatetimeIndex(X_test['release_date']).day
X_test['year'] = pd.DatetimeIndex(X_test['release_date']).year

X_train.drop(['release_date'], axis=1, inplace=True)
X_test.drop(['release_date'], axis=1, inplace=True)


lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

stop_words = set(nltk.corpus.stopwords.words('english'))


tfidf = TfidfVectorizer(stop_words="english")

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

X_train["overview"] = X_train["overview"].apply(lambda x: tokenizer.tokenize(x.lower()))

X_test["overview"] = X_test["overview"].apply(lambda x: tokenizer.tokenize(x.lower()))


X_train['overview'] = X_train['overview'].apply(lambda x: ' '.join([PorterStemmer().stem(word) for word in x if word not in stop_words]))


X_test['overview'] = X_test['overview'].apply(lambda x: ' '.join([PorterStemmer().stem(word) for word in x if word not in stop_words]))

X_train_tfidf = tfidf.fit_transform(X_train["overview"])

X_test["overview"] = X_test["overview"].fillna("")

X_test_tfidf = tfidf.transform(X_test["overview"])


overview_array = np.concatenate((X_train_tfidf.toarray(), X_test_tfidf.toarray()))

overview_similarities = linear_kernel(overview_array, overview_array)


X_train['overview'] = overview_similarities[:len(X_train), :len(X_train)]
X_test['overview'] = overview_similarities[len(X_train):, :len(X_train)]


with open('tf-idf', 'wb') as f:
    pickle.dump(tfidf, f)


X_train.drop(['tagline', 'id', 'original_title'], axis=1, inplace=True)
X_test.drop(['tagline', 'id', 'original_title'], axis=1, inplace=True)


X_train.shape


# perform ANOVA f-test to select top features
f_scores, p_values = f_classif(X_train, y_train)
top_features_idx = f_scores.argsort()[::-1][:6]  # select top 8 features

# select top features and display correlation matrix heatmap
plt.subplots(figsize=(10, 5))
X_train = X_train.iloc[:, top_features_idx]
X_test = X_test.iloc[:, top_features_idx]
sns.heatmap(X_train.corr(), annot=True)
plt.show()

with open('topfeature.pkl', 'wb') as f:
    pickle.dump(top_features_idx, f)


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)


X_train.shape


Q1 = np.percentile(X_train, 25)
Q3 = np.percentile(X_train, 75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

X_train = X_train[(y_train > lower_limit) & (y_train < upper_limit)]
X_test = X_test[(y_test > lower_limit) & (y_test < upper_limit)]

y_train = y_train[(y_train > lower_limit) & (y_train < upper_limit)]

y_test = y_test[(y_test > lower_limit) & (y_test < upper_limit)]


rf = RandomForestClassifier()
n_estimators = [10, 100, 1000]
grid = dict(n_estimators=n_estimators)
rf_grid_search = GridSearchCV(estimator=rf, param_grid=grid, n_jobs=-1, scoring='accuracy',error_score=0)

svm = SVC()
kernel = ['poly', 'rbf', 'sigmoid']
C = [50, 10, 1.0, 0.1, 0.01]
gamma = ['scale']
grid = dict(kernel=kernel, C=C, gamma=gamma)
svm_grid_search = GridSearchCV(estimator=svm, param_grid=grid, n_jobs=-1, scoring='accuracy', error_score=0)

nb = GaussianNB()

knn = KNeighborsClassifier()
n_neighbors = range(1, 21, 2)
weights = ['uniform', 'distance']
metric = ['euclidean', 'manhattan', 'minkowski']
grid = dict(n_neighbors=n_neighbors,weights=weights,metric=metric)
knn_grid_search = GridSearchCV(estimator=knn, param_grid=grid, n_jobs=-1, scoring='accuracy',error_score=0)

# train models
rf_grid_search.fit(X_train, y_train)
svm_grid_search.fit(X_train, y_train)
nb.fit(X_train, y_train)
knn_grid_search.fit(X_train, y_train)

# save trained models
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_grid_search, f)

with open('svm_model.pkl', 'wb') as f:
    pickle.dump(svm_grid_search, f)

with open('naive_bayes_model.pkl', 'wb') as f:
    pickle.dump(nb, f)

with open('knn_model.pkl', 'wb') as f:
    pickle.dump(knn_grid_search, f)

# generate predictions using trained models
rf_preds = rf_grid_search.predict(X_test)
svm_preds = svm_grid_search.predict(X_test)
nb_preds = nb.predict(X_test)
knn_preds = knn_grid_search.predict(X_test)

# evaluate models
print("Random Forest Accuracy:", accuracy_score(y_test, rf_preds))
print(confusion_matrix(y_test, rf_preds))

print("SVM Accuracy:", accuracy_score(y_test, svm_preds))
print(confusion_matrix(y_test, svm_preds))

print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_preds))
print(confusion_matrix(y_test, nb_preds))

print("KNN Accuracy:", accuracy_score(y_test, knn_preds))
print(confusion_matrix(y_test, knn_preds))

# load trained models from saved pickle files
with open('random_forest_model.pkl', 'rb') as f:
    rf_loaded = pickle.load(f)

with open('svm_model.pkl', 'rb') as f:
    svm_loaded = pickle.load(f)

with open('naive_bayes_model.pkl', 'rb') as f:
    nb_loaded = pickle.load(f)

with open('knn_model.pkl', 'rb') as f:
    knn_loaded = pickle.load(f)


models = ['Random Forest', 'SVM', 'Naive Bayes', 'KNN']

rf_accuracy = accuracy_score(y_test, rf_preds)
svm_accuracy = accuracy_score(y_test, svm_preds)
nb_accuracy = accuracy_score(y_test, nb_preds)
knn_accuracy = accuracy_score(y_test, knn_preds)

accuracies = [rf_accuracy, svm_accuracy, nb_accuracy, knn_accuracy]


plt.bar(models, accuracies)
plt.title('Classification Accuracy')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0.5, 1) # Set y limit from 0.5 to 1
plt.show()

start_time = time.time()
rf_grid_search.fit(X_train, y_train)
rf_train_time = time.time() - start_time

start_time = time.time()
svm_grid_search.fit(X_train, y_train)
svm_train_time = time.time() - start_time

start_time = time.time()
nb.fit(X_train, y_train)
nb_train_time = time.time() - start_time

start_time = time.time()
knn_grid_search.fit(X_train, y_train)
knn_train_time = time.time() - start_time

# generate predictions and measure test time
start_time = time.time()
rf_preds = rf_grid_search.predict(X_test)
rf_test_time = time.time() - start_time

start_time = time.time()
svm_preds = svm_grid_search.predict(X_test)
svm_test_time = time.time() - start_time

start_time = time.time()
nb_preds = nb.predict(X_test)
nb_test_time = time.time() - start_time

start_time = time.time()
knn_preds = knn_grid_search.predict(X_test)
knn_test_time = time.time() - start_time

train_times = [rf_train_time, svm_train_time, nb_train_time, knn_train_time]

plt.bar(models, train_times)
plt.title('Total Training Time for Each Model')
plt.xlabel('Model')
plt.ylabel('Total Training Time (Seconds)')
plt.show()

# create bar graphs for test time
test_times = [rf_test_time, svm_test_time, nb_test_time, knn_test_time]

plt.bar(models, test_times)
plt.title('Total Test Time for Each Model')
plt.xlabel('Model')
plt.ylabel('Total Test Time (Seconds)')
plt.show()