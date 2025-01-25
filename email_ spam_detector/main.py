# main.py
from backend.preprocess import load_data, preprocess_data
from model import train_model, evaluate_model
import pickle

def main():
    # Load and preprocess data
    data = load_data('data\messages_dataset.csv')
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

if __name__ == "__main__":
    main()
