

import torch.nn.functional as F

import json
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
from openai import OpenAI

client = OpenAI(api_key="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")
import os




key_terms = [
    "C++",
    "syntax",
    "variables",
    "data types",
    "control structures",
    "loops",
    "functions",
    "recursion",
    "pointers",
    "arrays",
    "strings",
    "classes and objects",
    "inheritance",
    "polymorphism",
    "templates",
    "Standard Template Library (STL)",
    "iterators",
    "containers",
    "algorithms",
    "exception handling",
    "file I/O",
    "memory management",
    "dynamic allocation",
    "smart pointers",
    "lambda expressions",
    "concurrency",
    "threading",
    "debugging",
    "compilation",
    "linking",
    "data abstraction",
    "encapsulation",
    "friend functions",
    "operator overloading",
    "copy constructor",
    "assignment operator",
    "virtual functions",
    "abstract classes",
    "interfaces",
    "function templates",
    "class templates",
    "STL algorithms",
    "STL containers",
    "vector",
    "list",
    "map",
    "set",
    "queue",
    "stack",
    "deque",
    "bitset",
    "unordered_map",
    "unordered_set",
    "Constructors and Destructors",
    "Binary Trees",
    "Depth-First Search",
    "Sorting Algorithms",
    "Trees",
    "Tree Traversal",
    "Graph Traversals",
    "Data Structures",
    "Algorithms",
    "Object-Oriented Programming"
]
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



def get_context(query):
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

    return relevant_paragraphs


def get_first_block_index(problem):
    relevant_info = get_context(problem)
    # Adjusted to match the expected API structure
    messages = [
        {
            "role": "system",  # or "user" depending on the context
            "content": f'''        
You are a Tutorbot, an AI-powered chatbot designed to help students solve a problem.
For being a good Tutorbot, you should be able to break the problem into sequential subproblems.
Also, you should be able to predict possible incorrect student responses to each subproblem.
For the incorrect student responses, your job also involves providing necessary feedback to the student.
And ofcourse, for being a good Tutorbot, you should know the facts needed to answer the problem and most critically the solution to the problem.

Also, here's some information that can be useful to Tutorbot:
{relevant_info}

Create a Question about {problem}, now please provide the following:

1) A problem about the topic
2)Facts necessary to answer it,
3) Subproblems that the main problem can be broken down into, and
4) The final answer.
For each subproblem, generate a hint, one incorrect student response to the subproblem, and corresponding feedback to the student. Put all the output in the following JSON structure:
{{
    "SubProblems": [
            "Question": "..",
            "Answer": "..",
            "Hint": "..",
            "Incorrect Response": "..",
            "Feedback": ".."
    ],
    "Facts": [
        "..",
        ".."
    ],
    "Solution": ".."
}}.
 Now please provide subproblems with necessary hints, possible student incorrect responses, feedback, along with facts and solution for the problem.

'''
        }
    ]
    return messages






def generate_content_with_openai(prompt):
    try:

        response = client.chat.completions.create(
            model="gpt-4",
            messages=prompt,
            max_tokens = 2048,
            )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating content: {e}")
        return ""

# Main function to produce JSON data for each key term
def produce_prob_json():
    data = []
    for term in key_terms:
            for i in range(10):
                prompt = get_first_block_index(term)
                content = generate_content_with_openai(prompt)
                print(content)
                data.append({"term": term, "content": content})
    return data

if __name__ == '__main__':
    generated_data = produce_prob_json()
    with open('generated_content.json', 'w') as f:
        json.dump(generated_data, f, indent=4)
