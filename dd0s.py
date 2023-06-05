import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm

# Завантаження навченої моделі
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Витягнення ознак з файлу
def extract_features(file_path):
    with open(file_path, 'r', errors='ignore') as file:
        content = file.read()
    return content

# Класифікація файлу
def classify_file(model, features):
    vectorizer = TfidfVectorizer()
    features_vector = vectorizer.fit_transform([features])
    prediction = model.predict(features_vector)
    return prediction[0]

# Головна функція
def main():
    # Шлях до навченої моделі
    model_path = 'path/to/trained/model.pkl'

    # Шлях до файлу для аналізу
    file_path = 'path/to/file.exe'

    # Завантаження моделі
    model = load_model(model_path)

    # Витягнення ознак з файлу
    features = extract_features(file_path)

    # Класифікація файлу
    prediction = classify_file(model, features)

    if prediction == 1:
        print("Файл є потенційно шкідливим.")
    else:
        print("Файл безпечний.")

if __name__ == "__main__":
    main()
