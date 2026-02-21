# ner_extractor.py

import re
from typing import List, Dict
import string
from collections import defaultdict
import time

import spacy
import yake
from nltk.corpus import stopwords
import nltk
from spacy.tokens import Doc, Span
# You may need to install this: pip install python-dateutil
from dateutil.parser import parse

nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))
STOPWORDS.add('pg_no')
STOPWORDS.add('cnk')

YAKE_MAX_NGRAM_SIZE = 3
YAKE_NUM_KEYWORDS = 40
YAKE_DEDUP_THRESHOLD = 0.9

# --- Use a faster spaCy model ---
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")
print("Model loaded.")


# ----------------------------
# Preprocessing
# ----------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"\n\d+\s*\n", " ", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ----------------------------
# Helpers
# ----------------------------
def normalize_span(span: Span) -> str:
    """Normalizes a spaCy span for general text keywords."""
    noun_tokens = [
        token.lemma_.lower()
        for token in span
        if token.pos_ in {"NOUN", "PROPN"}
    ]
    if not noun_tokens: return ""
    while noun_tokens and noun_tokens[0] in STOPWORDS: noun_tokens.pop(0)
    while noun_tokens and noun_tokens[-1] in STOPWORDS: noun_tokens.pop()
    if not noun_tokens: return ""
    normalized = " ".join(noun_tokens)
    normalized = normalized.translate(str.maketrans("", "", string.punctuation))
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()

### NEW FUNCTION ###
def normalize_date(span: Span) -> str:
    """Normalizes a spaCy DATE entity to YYYY-MM-DD format."""
    try:
        # dateutil.parser is very robust at parsing various date formats
        dt = parse(span.text)
        return dt.strftime("%Y-%m-%d")
    except (ValueError, OverflowError):
        # Return empty if it's not a parsable date
        return ""

### NEW FUNCTION ###
def normalize_number(span: Span) -> str:
    """Normalizes and validates a spaCy CARDINAL/QUANTITY entity."""
    # Remove commas from numbers like "1,000"
    num_text = span.text.replace(",", "").strip()
    if num_text.isdigit():
        # Keep numbers that are likely to be years or other significant values
        if len(num_text) >= 4 or (10 <= int(num_text) <= 999):
             return num_text
    return "" # Discard less significant numbers (e.g., single digits)


def post_process_keyword(kw: str) -> str:
    """Cleans a normalized keyword by removing leading single-character stopwords."""
    tokens = kw.split()
    if len(tokens) > 1 and len(tokens[0]) == 1 and tokens[0] in STOPWORDS:
        return " ".join(tokens[1:])
    return kw


def is_valid_keyword(kw: str) -> bool:
    """Validates a keyword against a set of cleaning rules."""
    kw = kw.strip()
    if not kw: return False
    
    # NOTE: The aggressive `isdigit()` check has been removed.
    # We now rely on the more intelligent normalize_number function.

    tokens = kw.split()
    if all(t in STOPWORDS or t in string.punctuation for t in tokens):
        return False
    if len("".join(tokens)) <= 2:
        return False
    return True

# ----------------------------
# Extractor Functions
# ----------------------------
def extract_yake(text: str) -> List[str]:
    kw_extractor = yake.KeywordExtractor(
        lan="en", n=YAKE_MAX_NGRAM_SIZE, dedupLim=YAKE_DEDUP_THRESHOLD,
        top=YAKE_NUM_KEYWORDS, features=None
    )
    return [kw for kw, score in kw_extractor.extract_keywords(text)]

def extract_names_regex(text: str) -> List[str]:
    candidates = set()
    proper_case = re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b", text)
    candidates.update(proper_case)
    all_caps = re.findall(r"\b(?:[A-Z]{2,}(?:\s|$)){2,}\b", text)
    candidates.update([a.strip() for a in all_caps])
    return list(candidates)

### MODIFIED AND IMPROVED FUNCTION ###
def extract_spacy(doc: Doc) -> List[str]:
    """
    Extracts entities using spaCy and routes them to the correct
    normalization function based on their entity label.
    """
    results = set()
    
    # Process named entities with specific logic
    for ent in doc.ents:
        label = ent.label_
        if label == "DATE":
            normalized = normalize_date(ent)
            if normalized: results.add(normalized)
        elif label in ["CARDINAL", "QUANTITY", "MONEY"]:
            normalized = normalize_number(ent)
            if normalized: results.add(normalized)
        elif label in ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT"]:
            # Use the general text normalization for these
            normalized = normalize_span(ent)
            if normalized: results.add(normalized)

    # Process noun chunks for more general keywords
    for chunk in doc.noun_chunks:
        # Avoid double-processing something that was already an entity
        if chunk.text not in [e.text for e in doc.ents]:
             normalized = normalize_span(chunk)
             if normalized: results.add(normalized)
             
    return list(results)


# ----------------------------
# Deduplication
# ----------------------------
def deduplicate_keywords(keywords: List[str]) -> List[str]:
    keywords.sort(key=len, reverse=True)
    final_keywords = []
    superstrings = set()
    for kw in keywords:
        if not any(kw in s for s in superstrings):
            final_keywords.append(kw)
            superstrings.add(kw)
    return final_keywords


# ----------------------------
# Unified Extractors
# ----------------------------
def extract_keywords(text: str) -> List[str]:
    clean_doc_text = clean_text(text)
    doc = nlp(clean_doc_text)

    # Step 1: Extract candidates. spaCy is now the primary, intelligent source.
    spacy_keywords = extract_spacy(doc)
    yake_keywords = []#extract_yake(clean_doc_text)
    regex_names = extract_names_regex(clean_doc_text)
    
    # Create a combined set of raw text from YAKE and Regex for normalization
    other_raw_keywords = set(yake_keywords + regex_names)

    # Step 2: Normalize and post-process the other raw candidates
    normalized_and_processed = set(spacy_keywords) # Start with the already-processed spaCy keywords
    for raw_kw in other_raw_keywords:
        kw_doc = nlp(raw_kw)
        normalized = normalize_span(kw_doc[:])
        post_processed = post_process_keyword(normalized)
        if post_processed:
            normalized_and_processed.add(post_processed)

    # Step 3: Final validation and deduplication
    validated_keywords = [kw for kw in normalized_and_processed if is_valid_keyword(kw)]
    final_keywords = deduplicate_keywords(validated_keywords)
    
    return final_keywords


# ----------------------------
# Keyword to Chunk Mapping
# ----------------------------
def map_keywords_to_chunks(chunks: List[str]) -> Dict[str, List[str]]:
    print(f"Processing {len(chunks)} chunks...")
    keyword_map = defaultdict(set)
    start_time = time.time()
    for i, chunk in enumerate(chunks):
        if (i + 1) % 10 == 0:
                print(f"  - Processing chunk {i+1}/{len(chunks)}")
        keywords_in_chunk = extract_keywords(chunk)
        for kw in keywords_in_chunk:
            keyword_map[kw].add(chunk)
    final_map = {kw: list(chunk_set) for kw, chunk_set in keyword_map.items()}
    end_time = time.time()
    print(f"\nNER complete in {end_time - start_time:.2f} seconds.")
    return final_map


# ----------------------------
# File Readers
# ----------------------------
def extract_keywords_from_document(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return extract_keywords(text)


def read_chunks_from_file(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        chunks = [line.strip() for line in f if line.strip()]
    return chunks


# ----------------------------
# Example Usage
# ----------------------------
if __name__ == "__main__":
    print("--- Running on a single document ---")
    single_file_path = "data/chunks.txt"
    try:
        keywords = extract_keywords_from_document(file_path=single_file_path)
        print(f"Extracted {len(keywords)} final keywords/entities from '{single_file_path}':\n")
        print(keywords[:15])
    except FileNotFoundError:
        print(f"Error: The file '{single_file_path}' was not found.")

    print("\n" + "="*50 + "\n")

    print("--- Running on a multi-chunk document ---")
    multi_chunk_file_path = "data/multi_chunks.txt"
    try:
        chunks = read_chunks_from_file(multi_chunk_file_path)
        keyword_to_chunks_map = map_keywords_to_chunks(chunks)
        print(f"\nCreated a map with {len(keyword_to_chunks_map)} unique keywords.")
        print("Example mappings (keyword -> number of chunks):")
        sorted_map = sorted(keyword_to_chunks_map.items(), key=lambda item: len(item[1]), reverse=True)
        for kw, chunk_list in sorted_map[:10]:
            print(f"- '{kw}': found in {len(chunk_list)} chunks")
    except FileNotFoundError:
        print(f"Error: The file '{multi_chunk_file_path}' was not found.")
        print("Please create a file at that location with one text chunk per line.")