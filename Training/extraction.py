import spacy
import pandas as pd
import fitz  # PyMuPDF
from collections import Counter

# Initialize spaCy
nlp = spacy.load("en_core_web_sm")


# Step 1: Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# def process_chunk(text):
#     """
#     Process a chunk of text: tokenize, remove stop words and punctuation, and lemmatize.
#     Returns a list of processed tokens.
#     """
#     doc = nlp(text)
#     tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
#     return tokens

# def extract_keywords_in_chunks(text, chunk_size=100000, n_keywords=50):
#     """
#     Extract keywords from text by processing it in chunks.
#     """
#     # Split the text into chunks
#     chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
#     # Initialize a Counter object to keep track of term frequencies across chunks
#     term_freq = Counter()
    
#     # Process each chunk
#     for chunk in chunks:
#         chunk_tokens = process_chunk(chunk)
#         term_freq.update(chunk_tokens)
    
#     # Get the most common terms
#     keywords = term_freq.most_common(n_keywords)
    
#     return [keyword for keyword, _ in keywords]

# Placeholder for text extraction from your PDF



# Step 3: Advanced Text Analysis with spaCy


def analyze_text_with_spacy_in_chunks(text, terms, chunk_size=1000000):
    definitions = []
    # Process the text in chunks
    for start in range(0, len(text), chunk_size):
        end = start + chunk_size
        doc = nlp(text[start:end])
        
        for term in terms:
            for token in doc:
                if token.lemma_.lower() == term.lower():
                    sentence = token.sent.text
                    definitions.append({"Term": term, "Definition": sentence})
                      # Assuming only one definition per term is needed

    return definitions

# Step 3: Write to CSV
def write_definitions_to_csv(definitions, csv_path):
    df = pd.DataFrame(definitions)
    df.to_csv(csv_path, index=False)

# Main Workflow
pdf_path = "/Users/andrewlanpouthakoun/Library/Mobile Documents/com~apple~CloudDocs/Stanford/Quizzem/Training/Reader-Beta-2012.pdf"  # Update with the actual path to your PDF
csv_path = "definitions_spreadsheet.csv"
  # Add more terms as needed

text = extract_text_from_pdf(pdf_path)

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
print(key_terms)

definitions = analyze_text_with_spacy_in_chunks(text, key_terms)
write_definitions_to_csv(definitions, csv_path)

print(f"Definitions extracted and saved to {csv_path}")
