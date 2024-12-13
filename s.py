import gensim
from gensim.models import KeyedVectors

import gensim.downloader as api

# Download the pre-trained Word2Vec model from Google (takes some time)
# This downloads a large file, so be patient.
word2vec_model_path = api.load("word2vec-google-news-300", return_path=True)

# Load the Word2Vec model using gensim
word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)

# Example: Get the word embedding for the word "king"
word_embedding = word2vec_model["king"]

# Print the dimensionality of the word embedding
print("Dimensionality of word embedding:", len(word_embedding))

# Example: Get the most similar words to "king"
similar_words = word2vec_model.most_similar("king")
print("Words most similar to 'king':", similar_words)

# Example: Calculate the similarity between two words
similarity_score = word2vec_model.similarity("king", "queen")
print("Similarity between 'king' and 'queen':", similarity_score)

# Example: Calculate the vector representing the combination of words "king" and "man" minus "woman"
result_vector = word2vec_model.most_similar(positive=["king", "man"], negative=["woman"], topn=1)
print("Vector representation of 'king - man + woman':", result_vector)