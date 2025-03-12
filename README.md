# Text Classification using NLP, Embeddings, and Transformers

This project focuses on **text classification** by processing natural language data using **embeddings** and **transformers** before training a neural network. The goal is to classify each text into a predefined category that represents its content type.

## Description

Text classification is a key task in Natural Language Processing (NLP). In this project, two different approaches are implemented:

1. **Embedding-based Neural Network**:  
   - Text is preprocessed and transformed into word embeddings.  
   - A **fully connected neural network** is trained using a **20,000-dimensional embedding layer** followed by a **dense layer with 500 neurons**.  

2. **Transformer-based Model**:  
   - The text is processed using **transformers** to generate contextual embeddings.  
   - The resulting embeddings are passed to a **dense layer with 20 neurons** for classification.  

Both approaches aim to optimize classification accuracy while exploring the differences in feature representation between traditional embeddings and transformer-generated embeddings.

## Techniques Used

- **Natural Language Processing (NLP)**:
  - Tokenization
  - Text cleaning and preprocessing
  - Stopword removal
  - Word embeddings for vector representation
  
- **Embeddings-based Model**:
  - Uses a **20,000-dimensional embedding layer**
  - Includes a **dense layer with 500 neurons**
  - Trained with categorical classification loss

- **Transformer-based Model**:
  - Utilizes a **pre-trained transformer model** for feature extraction
  - A **dense layer with 20 neurons** processes the transformer embeddings
  - Optimized for text classification using deep learning techniques

## Libraries Used

- **Pandas**: Data manipulation and text handling  
- **NumPy**: Numerical computations  
- **NLTK / spaCy**: Text preprocessing (tokenization, stopword removal, etc.)  
- **TensorFlow / PyTorch**: Neural network implementation  
- **Transformers (Hugging Face)**: Pre-trained transformer models  
- **Scikit-learn**: Model evaluation and performance metrics  


