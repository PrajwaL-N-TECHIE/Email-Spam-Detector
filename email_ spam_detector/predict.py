# predict.py
import pickle

def predict_spam(input_text):
    # Load the trained model and vectorizer
    with open('spam_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    
    with open('vectorizer.pkl', 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)
    
    # Transform input text using the vectorizer
    input_vect = vectorizer.transform([input_text])
    
    # Predict the result
    prediction = model.predict(input_vect)
    return "Spam" if prediction[0] == 1 else "Not Spam"

if __name__ == "__main__":
    # Example usage
    user_input = input("Enter an email message to classify: ")
    result = predict_spam(user_input)
    print(f"The message is classified as: {result}")
