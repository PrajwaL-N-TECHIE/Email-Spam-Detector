# model.py
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from backend.preprocess import load_data, preprocess_data

def train_model(X_train, y_train):
    # Initialize and train Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    # Predict and evaluate model performance
    predictions = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Classification Report:")
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    # Load and preprocess data
    data = load_data('data/spam.csv')
    X_train, X_test, y_train, y_test, vectorizer = preprocess_data(data)
    
    # Train and evaluate the model
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    
    # Save the model and vectorizer
    with open('spam_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    
    with open('vectorizer.pkl', 'wb') as vec_file:
        pickle.dump(vectorizer, vec_file)
    
    print("Model and vectorizer saved successfully!")
