import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from flask import Flask, request, render_template, session

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

def preprocess_text(text):
    if pd.isna(text) or isinstance(text, float):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text.strip()

def load_and_preprocess_data(path):
    df = pd.read_csv(path)
    
    original_count = len(df)
    df = df.dropna(subset=['title', 'text'])
    cleaned_count = len(df)
    
    df['combined_text'] = df['title'].str.cat(df['text'], sep=' ')
    df['cleaned_text'] = df['combined_text'].apply(preprocess_text)
    
    df['word_count'] = df['cleaned_text'].apply(lambda x: len(x.split()))
    df['char_count'] = df['cleaned_text'].apply(len)
    
    dataset_report = {
        'original_samples': original_count,
        'cleaned_samples': cleaned_count,
        'fake_percentage': df['label'].mean() * 100,
        'avg_word_count': df['word_count'].mean(),
        'avg_char_count': df['char_count'].mean()
    }
    
    return df, dataset_report

def train_model(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], df['label'], test_size=0.2, random_state=42
    )
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, solver='sag', n_jobs=-1),
    'Naive Bayes': MultinomialNB(),
    'SGD Classifier': SGDClassifier(
        loss='log_loss', 
        max_iter=1000,
        tol=1e-3,
        random_state=42,
        n_jobs=-1
    )
    }
    
    model_reports = {}
    best_accuracy = 0
    best_model = None
    
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred, output_dict=True)
        
        cv_scores = cross_val_score(model, X_train_vec, y_train, cv=5)
        
        model_reports[name] = {
            'accuracy': accuracy,
            'confusion_matrix': cm.tolist(),
            'classification_report': cr,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
    
    return best_model, vectorizer, model_reports

def predict_news(text, model, vectorizer):
    cleaned_text = preprocess_text(text)
    text_vec = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vec)[0]
    probability = model.predict_proba(text_vec)[0].max()
    return {
        'prediction': 'fake' if prediction == 1 else 'real',
        'confidence': round(probability * 100, 2),
        'processed_text': cleaned_text
    }

NEWS_PATH = "news.csv"
df, dataset_report = load_and_preprocess_data(NEWS_PATH)
model, vectorizer, model_reports = train_model(df)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    history = session.get('history', [])
    
    if request.method == 'POST':
        news_text = request.form['news_text']
        if len(news_text.strip()) < 50:
            return render_template('index.html', error="Please enter at least 50 characters")
        
        result = predict_news(news_text, model, vectorizer)
        prediction_result = {
            'text': news_text[:200] + '...' if len(news_text) > 200 else news_text,
            'processed_text': result['processed_text'][:200] + '...',
            'result': result['prediction'],
            'confidence': result['confidence']
        }
        
        history = [prediction_result] + history[:4]
        session['history'] = history
    
    return render_template('index.html', 
                         prediction=prediction_result,
                         history=history,
                         dataset_report=dataset_report,
                         model_reports=model_reports)

if __name__ == "__main__":
    app.run()