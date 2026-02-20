import math
from collections import defaultdict

def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))


def magnitude(v):
    return math.sqrt(sum(x * x for x in v))


def cosine_similarity(a, b):
    if magnitude(a) == 0 or magnitude(b) == 0:
        return 0
    return dot_product(a, b) / (magnitude(a) * magnitude(b))


def euclidean_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def manhattan_distance(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))

def cosine_search(query_vec, documents):
    scores = []
    for doc_id, vec in documents.items():
        score = cosine_similarity(query_vec, vec)
        scores.append((doc_id, score))
    return sorted(scores, key=lambda x: x[1], reverse=True)

def euclidean_search(query_vec, documents):
    scores = []
    for doc_id, vec in documents.items():
        dist = euclidean_distance(query_vec, vec)
        scores.append((doc_id, dist))
    return sorted(scores, key=lambda x: x[1])


def manhattan_search(query_vec, documents):
    scores = []
    for doc_id, vec in documents.items():
        dist = manhattan_distance(query_vec, vec)
        scores.append((doc_id, dist))
    return sorted(scores, key=lambda x: x[1])

def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0


def jaccard_search(query, documents):
    query_set = set(query.lower().split())
    scores = []
    for doc_id, text in documents.items():
        doc_set = set(text.lower().split())
        score = jaccard_similarity(query_set, doc_set)
        scores.append((doc_id, score))
    return sorted(scores, key=lambda x: x[1], reverse=True)

def build_inverted_index(documents):
    index = defaultdict(list)
    for doc_id, text in documents.items():
        for word in text.lower().split():
            index[word].append(doc_id)
    return index


def keyword_search(query, index):
    return index.get(query.lower(), [])

if __name__ == "__main__":

    vector_docs = {
        "doc1": [1, 2, 3],
        "doc2": [2, 3, 4],
        "doc3": [10, 10, 10]
    }

    query_vector = [1, 2, 2]

    print("Cosine Search:")
    print(cosine_search(query_vector, vector_docs))
    print()

    print("Euclidean Search:")
    print(euclidean_search(query_vector, vector_docs))
    print()

    print("Manhattan Search:")
    print(manhattan_search(query_vector, vector_docs))
    print()

    text_docs = {
        "doc1": "machine learning is fun",
        "doc2": "deep learning and ai",
        "doc3": "python programming language"
    }

    print("Jaccard Search:")
    print(jaccard_search("learning ai", text_docs))
    print()

    index = build_inverted_index(text_docs)
    print("Keyword Search (learning):")
    print(keyword_search("learning", index))
