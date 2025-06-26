# Search Engine with NLTK

This project implements a simple search engine using only Python Standard Library and NLTK. It supports lemmatization, normalization, abbreviation handling, positional inverted indexing, and edit-distance-based spelling correction.

## 📘 Project Overview

- **Language**: Python 3
- **Libraries**: Only `nltk`, `pickle`, and Python built-ins
- **Core Functions**:
  - Inverted index construction with positional info
  - Search with proximity-based ranking
  - Contextual line retrieval for matching terms
  - Spelling correction (edit distance ≤ 2)

## 🗃️ Dataset

- **Source**: Provided 1,000 text documents (as individual numbered files) under `data/`
- Example files: `1`, `5`, `6`

## 🛠️ File Structure

```text
search-engine-nltk/
├── data/                         # Contains 1000 documents (e.g., 1, 5, ..., 1000)
├── index.py                     # Builds the positional inverted index
├── search.py                    # Query processor with ranking and correction
├── COMP6714 2024T3 Project Specification.pdf
```

## ⚙️ How to Run

### 1. Indexing

```bash
python3 index.py data indexes
```

- Creates `indexes/index.pkl` and `indexes/docs_path.txt`.

### 2. Searching

#### Interactive Mode

```bash
python3 search.py indexes
```

#### Piped Input

```bash
echo "economic policy" | python3 search.py indexes
```

#### Special Query (`>`)

To show contextual lines instead of just document IDs:

```bash
echo "> global market" | python3 search.py indexes
```

## 🔍 Features

- **Preprocessing**: 
  - Lemmatization (verbs and nouns)
  - Removal of punctuation, abbreviations, and leading zeros
- **Spelling Correction**:
  - Based on Jaccard distance over bigrams
  - Matches first letter and selects closest edit distance
- **Ranking**:
  - Based on shortest span of matching terms in document
- **Contextual Line Display**:
  - Returns the line(s) from document containing query terms (for `> query`)

## 📄 Project Specification

Refer to `COMP6714 2024T3 Project Specification.pdf` for detailed requirements and evaluation criteria.