

import re

def basic_tokenize(text):
    """Split text by spaces"""
    return text.split()

def regex_tokenize(text):
    """Extract words using regex (removes punctuation)"""
    return re.findall(r'\b\w+\b', text)

def nltk_tokenize(text):
    """Tokenize using NLTK library"""
    import nltk
    from nltk.tokenize import word_tokenize

    nltk.download('punkt', quiet=True)
    return word_tokenize(text)

if __name__ == "__main__":
    text = "Machine learning is amazing! Isn't it?"

    print("Original Text:")
    print(text)
    print()

    print("Basic Tokenization:")
    print(basic_tokenize(text))
    print()

    print("Regex Tokenization:")
    print(regex_tokenize(text))
    print()

    print("NLTK Tokenization:")
    print(nltk_tokenize(text))
