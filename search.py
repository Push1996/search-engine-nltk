import os
import sys
import re
from nltk.stem import PorterStemmer
from collections import defaultdict
import pickle
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.metrics.distance import jaccard_distance
from nltk.util import ngrams

ps = PorterStemmer()

def remove_duplicates(lst):
    seen = set()
    result = []
    for num in lst:
        if num not in seen:
            seen.add(num)
            result.append(num)
    return result

def pairwise_distances(query_tokens, calc_positions):
    distances = []
    for i in range(len(query_tokens) - 1):
        distance = abs(calc_positions[i + 1] - calc_positions[i])
        if calc_positions[i + 1] > calc_positions[i]:
            distance -= 1
        distances.append(distance)
    return sum(distances)

def shortest_span_dp(query_tokens, inverted_index, doc):
    posting_lists = [inverted_index[token][doc]['positions'] for token in query_tokens]
    dp = [[float('inf')] * len(posting_lists[i]) for i in range(len(query_tokens))]
    path = [[None] * len(posting_lists[i]) for i in range(len(query_tokens))]

    for j in range(len(posting_lists[0])):
        dp[0][j] = 0

    for i in range(1, len(query_tokens)):
        for j in range(len(posting_lists[i])):
            min_distance = float('inf')
            min_index = None
            for k in range(len(posting_lists[i - 1])):
                distance = dp[i - 1][k] + abs(posting_lists[i][j] - posting_lists[i - 1][k] - 1)
                if distance < min_distance:
                    min_distance = distance
                    min_index = k
            dp[i][j] = min_distance
            path[i][j] = min_index

    min_distance_sum = float('inf')
    best_position_index = None
    for j in range(len(posting_lists[-1])):
        if dp[-1][j] < min_distance_sum:
            min_distance_sum = dp[-1][j]
            best_position_index = j

    best_positions = [None] * len(query_tokens)
    current_index = best_position_index
    for i in range(len(query_tokens) - 1, -1, -1):
        best_positions[i] = posting_lists[i][current_index]
        current_index = path[i][current_index]

    return min_distance_sum, best_positions

def adjust_zeros(s):
    return "0" if all(c == '0' for c in s) else s.lstrip("0")

def preprocess(token):
    lemmatizer = WordNetLemmatizer()
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

    return token

def load_indexes(folder):
    with open(os.path.join(folder, 'index.pkl'), 'rb') as handle:
        return pickle.load(handle)

def load_docs_path(folder):
    with open(os.path.join(folder, 'docs_path.txt'), 'r') as f:
        return f.read().strip()

def spell_correct(word, correct_words):
    candidates = [(jaccard_distance(set(ngrams(word, 2)), set(ngrams(w, 2))), w) 
                  for w in correct_words if w[0] == word[0]]
    return sorted(candidates, key=lambda val: val[0])[0][1] if candidates else word

def build_correct_words(index):
    correct_words = set()
    for term in index:
        correct_words.add(term)
    return list(correct_words)

def rank_results(query, index):
    terms = query
    doc_scores = defaultdict(int)
    best_dict = defaultdict(list)
    search_doc_result = set(index[terms[0]].keys())
    
    for token in terms[1:]:
        search_doc_result = search_doc_result.intersection(set(index[token].keys()))

    for doc in search_doc_result:
        doc_scores[doc], best_dict[doc] = shortest_span_dp(terms, index, doc)

    return sorted(doc_scores.keys(), key=lambda x: doc_scores[x]), best_dict

def get_lines(filename, position):
    with open(filename, 'r') as f:
        lines = f.readlines()
        result_lines = lines[position]

    return result_lines.strip('\n')

def process_query(query, index, correct_words, docs_path):
    if query.startswith(">"):
        term_list = query[2:].strip('\n').split(' ')
        term_list = [preprocess(spell_correct(preprocess(x), correct_words)) if preprocess(x) not in index else preprocess(x) for x in term_list]

        if len(term_list) == 1:
            doc_list = [int(x) for x in index[term_list[0]].keys()]
            doc_list.sort()
            for doc in doc_list:
                print('> ' + str(doc))
                print(get_lines(os.path.join(docs_path, str(doc)), index[term_list[0]][doc]['lines'][0]))
        elif len(term_list) >= 2:
            rank_list, mini_position = rank_results(term_list, index)
            for doc in rank_list:
                print('> ' + str(doc))
                tmp_list = []
                for i in range(len(term_list)):
                    line_num = index[term_list[i]][doc]['lines'][index[term_list[i]][doc]['positions'].index(mini_position[doc][i])]
                    tmp_list.append(line_num)
                new_list = remove_duplicates(tmp_list)
                new_list.sort()
                for x in new_list:
                    print(get_lines(os.path.join(docs_path, str(doc)), x))
    else:
        term_list = query.strip('\n').split(' ')
        term_list = [preprocess(spell_correct(preprocess(x), correct_words)) if preprocess(x) not in index else preprocess(x) for x in term_list]
        if len(term_list) == 1:
            doc_list = [int(x) for x in index[term_list[0]].keys()]
            doc_list.sort()
            for doc in doc_list:
                print(doc)
        elif len(term_list) >= 2:
            rank_list, _ = rank_results(term_list, index)
            for doc in rank_list:
                print(doc)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 search.py [folder-of-indexes]")
        sys.exit(1)

    index_folder = sys.argv[1]
    index = load_indexes(index_folder)
    correct_words = build_correct_words(index)
    docs_path = load_docs_path(index_folder)  # Load docs_path from docs_path.txt

    if os.isatty(sys.stdin.fileno()):
        while True:
            try:
                query = input()
                process_query(query, index, correct_words, docs_path)
            except EOFError:
                break
    else:
        for query in sys.stdin:
            process_query(query, index, correct_words, docs_path)

if __name__ == "__main__":
    main()
