from flask import Flask, request, render_template
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# Load the saved model, tokenizer, and MultiLabelBinarizer
model = load_model('model_checkpoint.keras')
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)
with open('mlb.pkl', 'rb') as file:
    mlb = pickle.load(file)

# Flask app setup
app = Flask(__name__)



@app.route('/predict_genre', methods=['POST'])
def predict_genre_route():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    if file and allowed_file(file.filename):
        # Save the uploaded file temporarily
        file_path = 'temp.srt'
        file.save(file_path)
        
        # Predict genre from the uploaded SRT file
        genre_prediction = predict_genre(file_path)
        return f"The predicted genres are: {genre_prediction}"

    return 'Invalid file type'

def allowed_file(filename):
    return filename.endswith('.srt')

def predict_genre(srt_file_path):
    # Preprocess the SRT file to tokenize and pad the sequence
    padded_sequence = preprocess_srt(srt_file_path, tokenizer, max_length=30)  # Ensure max_length matches with the training
    
    # Predict genre (multi-label)
    prediction = model.predict(padded_sequence)
    
    # Get the genre labels where prediction > 0.5 (binary threshold for multi-label classification)
    predicted_labels = np.where(prediction > 0.3, 1, 0)
    
    # Decode the labels to get the genre names
    genres = mlb.inverse_transform(predicted_labels)
    return ", ".join(genres[0])

def preprocess_srt(file_path, tokenizer, max_length):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Remove timestamps and keep only text
    text = re.sub(r'\d+\s+\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', '', content)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\n+', ' ', text)  # Replace new lines with spaces

    # Tokenize and pad the text
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequences, maxlen=max_length)
    
    return padded_sequence

if __name__ == '__main__':
    app.run(debug=True)
