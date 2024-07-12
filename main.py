import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from joblib import dump

# Ensure necessary NLTK resources are available
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    lm = WordNetLemmatizer()
    review = re.sub('[^a-zA-Z0-9]', ' ', text)
    review = review.lower().split()
    review = [word for word in review if word not in stopwords.words('english')]
    review = [lm.lemmatize(word) for word in review]
    processed_text = ' '.join(review)
    return processed_text

# Load dataset and preprocess text
df = pd.read_csv("dataset/SMSSpamCollection.txt", sep='\t', names=['Label', 'Msg'])
df['Msg'] = df['Msg'].apply(preprocess_text)

x_train, x_test, y_train, y_test = train_test_split(df['Msg'], df['Label'], test_size=0.2, random_state=10)

models = {
    'Logistic Regression': LogisticRegression(solver='liblinear'),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Multinomial NB': MultinomialNB()
}

parameters = {
    'Logistic Regression': {'tfidf__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': [True, False], 'model__C': [0.1, 1, 10]},
    'SVM': {'tfidf__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': [True, False], 'model__C': [0.1, 1, 10]},
    'Random Forest': {'tfidf__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': [True, False], 'model__n_estimators': [50, 100, 200]},
    'Gradient Boosting': {'tfidf__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': [True, False], 'model__n_estimators': [50, 100, 200]},
    'Multinomial NB': {'tfidf__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': [True, False], 'model__alpha': [0.01, 0.1, 1]}
}

for name, model in models.items():
    pipeline = Pipeline([('tfidf', TfidfVectorizer()), ('model', model)])
    grid_search = GridSearchCV(pipeline, parameters[name], cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(x_train, y_train)
    dump(grid_search.best_estimator_, f"models/{name.replace(' ', '_').lower()}_model.joblib")

    y_pred = grid_search.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # Save metrics to CSV
    report_df = pd.DataFrame(report).transpose()
    report_df['accuracy'] = accuracy
    report_df.to_csv(f"reports/{name.replace(' ', '_').lower()}_report.csv", index=True)
    
    cm_df = pd.DataFrame(cm, index=['Actual_Ham', 'Actual_Spam'], columns=['Predicted_Ham', 'Predicted_Spam'])
    cm_df.to_csv(f"reports/{name.replace(' ', '_').lower()}_confusion_matrix.csv", index=True)

print("Model training and metrics saving complete.")
