import os
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from collections import defaultdict
import sys
import re
import pickle


def adjust_zeros(s):
    return "0" if all(c == '0' for c in s) else s.lstrip("0")

def preprocess(document):
    pattern = r"\b(?<!\d\.)[A-Za-z]+(?:'s)?(?:\.[A-Za-z]+\.?)*\b|\b(?<!\d\.)\d+(?:,\d{3})*(?!\.)\b"
    tokenizer = RegexpTokenizer(pattern)
    tokens = tokenizer.tokenize(document)

    lemmatizer = WordNetLemmatizer()
    preprocessed_tokens = []
    
    for token in tokens:
        token = token.lower()

        if '.' in token:
            token = token.replace('.', '')

        if ',' in token and token.replace(',', '').isdigit():
            token = token.replace(',', '')

        if '\'s' in token:
            token = token.replace('\'s', '')

        if token[0] == '0':
            token = adjust_zeros(token)

        token = lemmatizer.lemmatize(token, pos='v')
        token = lemmatizer.lemmatize(token, pos='n')
        
        preprocessed_tokens.append(token)

    return preprocessed_tokens

def create_index(docs_path):
    inverted_index = {}
    
    for doc_id in os.listdir(docs_path):
        with open(os.path.join(docs_path, doc_id), 'r', encoding='utf-8') as file:
            global_position = 0
            for line_num, line in enumerate(file, start=0):
                tokens = preprocess(line)
                for token in tokens:
                    if token not in inverted_index:
                        inverted_index[token] = {}
                    if int(doc_id) not in inverted_index[token]:
                        inverted_index[token][int(doc_id)] = {"lines": [], "positions": []}
                    
                    if line_num not in inverted_index[token][int(doc_id)]["lines"]:
                        inverted_index[token][int(doc_id)]["lines"].append(line_num)
                    
                    inverted_index[token][int(doc_id)]["positions"].append(global_position)
                    global_position += 1
                
    return inverted_index

def save_index_as_pickle(inverted_index, indexes_path, docs_path):
    if not os.path.exists(indexes_path):
        os.makedirs(indexes_path, exist_ok=True)

    with open(os.path.join(indexes_path, 'index.pkl'), 'wb') as f:
        pickle.dump(inverted_index, f)

    with open(os.path.join(indexes_path, 'docs_path.txt'), 'w') as f:
        f.write(docs_path)

def main():
    docs_path = sys.argv[1]
    indexes_path = sys.argv[2]

    inverted_index = create_index(docs_path)

    save_index_as_pickle(inverted_index, indexes_path, docs_path)


if __name__ == '__main__':
    main()
