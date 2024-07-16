# SMS Spam Detection

## Kaggle

Find my work on kaggle : [Kaggle link][https://www.kaggle.com/code/junioralive/top-10-algorithms-ham-or-spam-classifiers]

## Overview

This application leverages multiple machine learning models to accurately classify SMS messages as either spam or ham (non-spam). The application provides an interactive interface for users to input SMS text to receive instant predictions. It also includes a detailed analysis section showcasing the performance metrics of each deployed model.

## Features

- **Spam Prediction**: Users can input an SMS text and get predictions on whether the message is spam or ham.
- **Model Performance**: Displays detailed performance metrics for each model, including accuracy, classification reports, and confusion matrices.

## Models Used

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest Classifier
- Gradient Boosting Classifier
- Multinomial Naive Bayes

## Installation

To set up and run this application locally, follow the steps below:

### 1. Clone the Repository

```bash
git clone https://github.com/junioralive/sms-spam-detection.git
cd sms-spam-detection
```

### 2. Create and Activate a Virtual Environment

For Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

For macOS and Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Resources

Before running the application, download the required NLTK resources by executing the following Python commands:

```python
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
```

## Running the App

To run the app, use the following command in the project directory:

```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your web browser to interact with the application.

## Contributing

Contributions are welcome! Here are a few ways you can help improve the project:

- Report bugs.
- Propose new features.
- Submit pull requests for bug fixes or new functionalities.
- Improve documentation.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
