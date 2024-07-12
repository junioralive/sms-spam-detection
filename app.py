import streamlit as st
from joblib import load
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Load models
def load_models():
    models = {
        'Logistic Regression': load('models/logistic_regression_model.joblib'),
        'SVM': load('models/svm_model.joblib'),
        'Random Forest': load('models/random_forest_model.joblib'),
        'Gradient Boosting': load('models/gradient_boosting_model.joblib'),
        'Multinomial NB': load('models/multinomial_nb_model.joblib')
    }
    return models

models = load_models()

# Load metrics data
def load_metrics(model_name):
    report_path = f"reports/{model_name.replace(' ', '_').lower()}_report.csv"
    cm_path = f"reports/{model_name.replace(' ', '_').lower()}_confusion_matrix.csv"
    report_df = pd.read_csv(report_path, index_col=0)
    cm_df = pd.read_csv(cm_path, index_col=0)
    return report_df, cm_df

# Predict spam function
def predict_spam(text, model):
    nltk.download('stopwords')
    nltk.download('wordnet')
    lm = WordNetLemmatizer()
    review = re.sub('[^a-zA-Z0-9]', ' ', text)
    review = review.lower().split()
    review = [word for word in review if word not in stopwords.words('english')]
    review = [lm.lemmatize(word) for word in review]
    processed_text = ' '.join(review)
    
    prediction = model.predict([processed_text])[0]
    return "This Message is Spam" if prediction == 'spam' else "This Message is Ham"

# Streamlit application layout
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Spam/Ham Prediction", "Model Performance"])

    if page == "Spam/Ham Prediction":
        st.title("Spam SMS Detection")
        st.image("banner.jpg", use_column_width=True)
        st.write("Enter a SMS to predict if it's Spam or Ham")
        user_input = st.text_area("SMS Text", height=150)
        model_name = st.selectbox("Select Model", list(models.keys()))

        if st.button("Predict"):
            model = models[model_name]
            result = predict_spam(user_input, model)
            st.write("Prediction:", result)

    elif page == "Model Performance":
        st.title("Model Performance Metrics")
        model_name = st.selectbox("Select Model to View Performance", list(models.keys()))

        if model_name:
            report_df, cm_df = load_metrics(model_name)
            st.write(f"### Performance of {model_name}")
            st.write("**Classification Report**")
            st.dataframe(report_df.drop(columns=["support"]))
            
            accuracy = report_df.loc['accuracy', 'precision']
            st.write(f"**Accuracy**: {accuracy:.4f}")

            st.write("**Confusion Matrix**")
            plt.figure(figsize=(10, 7))
            sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix - {model_name}')
            st.pyplot(plt)

if __name__ == '__main__':
    main()
