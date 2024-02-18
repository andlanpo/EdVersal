import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import faiss
import numpy as np

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),
                                                                              min=1e-9)


# Read the dataframe
dataframe = pd.read_csv('/Users/andrewlanpouthakoun/Library/Mobile Documents/com~apple~CloudDocs/Stanford/Quizzem/Training/definitions_spreadsheet.csv')
paragraphs = dataframe['Definition'].tolist()
print(paragraphs[0])

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Tokenize paragraphs and compute embeddings
paragraph_embeddings = []

for index, paragraph in enumerate(paragraphs):
    #print(index)
    
    if isinstance(paragraph, str) and len(paragraph) > 10:
        encoded_input = tokenizer(paragraph, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        sentence_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
        normalized_embedding = F.normalize(sentence_embedding, p=2, dim=1)
        paragraph_embeddings.append(normalized_embedding.squeeze().numpy())

# Convert embeddings to a NumPy array
paragraph_embeddings = np.array(paragraph_embeddings)

# Use Faiss to store embeddings
index = faiss.IndexFlatIP(paragraph_embeddings.shape[1])
index.add(paragraph_embeddings)

print("Writing index")
# Save the index to a file
faiss.write_index(index, 'paragraph_index.faiss')

index = faiss.read_index('paragraph_index.faiss')

# Perform a query
query = 'What is recursion?'

# Tokenize and compute embedding for the query
encoded_query = tokenizer(query, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    query_output = model(**encoded_query)
query_embedding = mean_pooling(query_output, encoded_query['attention_mask'])
normalized_query_embedding = F.normalize(query_embedding, p=2, dim=1)

# Ensure query_embedding is a 2D array before search
normalized_query_embedding_2d = np.expand_dims(normalized_query_embedding.squeeze().numpy(), axis=0)

# Perform a search using Faiss
k = 5  # Number of nearest neighbors to retrieve
distances, indices = index.search(normalized_query_embedding_2d, k)

# Get the relevant paragraphs
relevant_paragraphs = [paragraphs[i] for i in indices[0]]

print(relevant_paragraphs)
