import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
import pickle

# Load the dataset
data = pd.read_csv('movies_genres.csv', delimiter='\t', quotechar='"')

# Separate plot and genres
texts = data['plot']
genres = data.drop(columns=['title', 'plot'])

# Convert the genre columns to a binary matrix (0/1)
mlb = MultiLabelBinarizer()
genres_binary = genres.apply(lambda row: row.index[row == 1].tolist(), axis=1)
genres_binary = mlb.fit_transform(genres_binary)

# Tokenize and pad the plot (text) column
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
max_length = max(len(seq) for seq in sequences)  # Find max sequence length for padding
padded_sequences = pad_sequences(sequences, maxlen=max_length)

# Define the model
vocab_size = len(tokenizer.word_index) + 1  # +1 because indexing starts from 1
embedding_dim = 100

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    LSTM(128, return_sequences=True),
    LSTM(64),
    Dense(64, activation='relu'),
    Dense(genres_binary.shape[1], activation='sigmoid')  # Sigmoid for multi-label classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, genres_binary, validation_split=0.2, epochs=10, batch_size=32)

# Save the model, tokenizer, and label encoder (MultiLabelBinarizer)
model.save('model_checkpoint.keras')

# Save tokenizer
with open('tokenizer.pkl', 'wb') as file:
    pickle.dump(tokenizer, file)

# Save MultiLabelBinarizer
with open('mlb.pkl', 'wb') as file:
    pickle.dump(mlb, file)

print("Model training complete and files saved!")
