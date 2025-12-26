import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# 1. The Data (Unstructured Text)

raw_data = [
    {"title": "Star Wars",  "plot": "space galaxy war empire rebels force"},
    {"title": "The Notebook", "plot": "love romance heart relationship couple"},
    {"title": "Iron Man",     "plot": "technology suit robot billionaire action"},
    {"title": "Wall-E",       "plot": "robot space future earth love"},
]

# Prepare data for Word2Vec (it needs a list of lists of words)
training_data = [d['plot'].split() for d in raw_data]

# 2. The Embedding Model (Creating Vectors)
# Training the Word2Vec model on the dataset.
# min_count=1 ensures we don't ignore words that appear only once.
# vector_size=10 means every word becomes a list of 10 numbers.
model = Word2Vec(sentences=training_data, vector_size=10, min_count=1, window=5)

print(f"Vector for 'space': {model.wv['space']}") 
# Output: [-0.012, 0.045, ..... ] (A list of floats representing 'space')


# 3. Helper Function (Sentence Embeddings)
# Word2Vec gives vectors for words. To search for a plot (sentence),
# We average the vectors of all words in that sentence.
def get_sentence_vector(sentence):
    words = sentence.lower().split()
    # Get the vector for each word if the model knows it
    word_vectors = [model.wv[w] for w in words if w in model.wv]
    
    if not word_vectors:
        return np.zeros(10) # Return empty vector if no words known
        
    # Average them together to get one vector for the whole sentence
    return np.mean(word_vectors, axis=0)

# 4. The Database
# We pre-calculate the vector for every movie in our database
vector_db = []

for movie in raw_data:
    vec = get_sentence_vector(movie['plot'])
    vector_db.append({"title": movie['title'], "vector": vec, "plot": movie['plot']})

# 5. The Search Engine (Cosine Similarity & RAG)

def search_movies(user_query):
    # Convert user query to vector (Encoding)
    query_vec = get_sentence_vector(user_query)
    
    # Compare query against every movie in DB
    results = []
    for entry in vector_db:
        # Calculate Cosine Similarity:
        # We reshape to (1, -1) because sklearn expects 2D arrays
        score = cosine_similarity([query_vec], [entry['vector']])[0][0]
        results.append((score, entry))
    
    # C. Sort by highest score (Ranking)
    results.sort(key=lambda x: x[0], reverse=True)
    return results[0] # Return the best match

# 6. The Example

# User searches for something related to "Space" and "Robots"
# Note: The word "future" is in Wall-E's plot, but not in our query. 
# The model might still find connection if trained on enough data.
query = "future robot technology" 

best_score, best_match = search_movies(query)

print("\n--- SEARCH RESULTS ---")
print(f"User Query: '{query}'")
print(f"Best Match: {best_match['title']}")
print(f"Confidence: {best_score:.4f}") # 1.0 is perfect match, -1.0 is opposite
print(f"Context: {best_match['plot']}")

# This "Context" would then be sent to ChatGPT/LLM to write the final answer.
