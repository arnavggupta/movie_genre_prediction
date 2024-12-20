# Movie Genre Prediction

This project explores the prediction of movie genres from subtitle scripts using deep learning techniques, specifically leveraging **Long Short-Term Memory (LSTM)** networks for multi-label classification.

---

## 📚 Overview

The primary goal of this project is to classify movies into multiple genres by analyzing their subtitles. The project involves the following key steps:

- **Data Format**: Subtitles in `.srt` format.
- **Text Preprocessing**: Tokenization, stopword removal, and sequence padding.
- **Feature Extraction**: Using Keras's `Tokenizer` to convert text into integer sequences, followed by multi-label binarization of genre labels.
- **Model Architecture**: 
  - LSTM layers for sequence modeling.
  - Dense layers for multi-label classification.
- **Evaluation**: The model's performance is evaluated on a dataset of movie subtitles, achieving accurate and reliable multi-label genre predictions.

---

## 🧪 Key Features

1. **Multi-Label Classification**: Predicts multiple genres for a single movie.
2. **Deep Learning Architecture**: Uses LSTM layers for effective sequence analysis.
3. **Text Preprocessing**: Efficiently handles noisy text data from subtitles.

---

## 🚀 Dataset

The dataset used for this project contains detailed information about movies, including their titles, subtitles, and genre labels. The data is organized in a tabular format with the following columns:

Title: The movie's title, including its release year.
Subtitle: A conversation of movie
Genres: Binary indicators (1 or 0) for each genre, such as Action, Comedy, Drama, etc., representing whether the movie belongs to a specific genre. You can access the dataset using the following link:

[Download Dataset](https://drive.google.com/file/d/10IH9FhKDpr_AELmlb0Fa9vzieYnU3jZV/view?usp=drivesdk)

---

## ⚙️ Future Improvements

- Experiment with additional deep learning architectures such as Transformers.
- Incorporate more sophisticated feature extraction techniques (e.g., Word2Vec, BERT).
- Extend the dataset to include more diverse subtitle sources.

---

## 👥 Team Members

- **Amon Sharma** - Roll No: 202251015  
- **Garv Arora** - Roll No: 202251048  
- **Arnav Gupta** - Roll No: 202251023  
- **Om Kumar** - Roll No: 202251081  

---

## 🛠️ Tools and Technologies

- **Programming Language**: Python
- **Libraries**:  Keras, NumPy, Pandas, Scikit-learn
- **Data Format**: `.srt` files

---

## 📝 Citation

If you use this work, please consider citing the project.  
Feel free to explore, use, and suggest improvements to this repository!
