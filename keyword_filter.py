from typing import Dict, List, Set

# --- Configuration ---
from config import FREQUENCY_THRESHOLD, KEEP_LIST

def filter_keys(
    key_chunk_map: Dict[str, List[str]],
    total_chunks: int
) -> Dict[str, List[str]]:

    """
    Filters a keyword-to-chunk map based on document frequency.

    Removes keywords that are too common (appear in a high percentage of chunks)
    while protecting essential keywords specified in a keep_list.

    Args:
        key_chunk_map: Dictionary mapping keywords to a list of chunks they appear in.
        total_chunks: The total number of chunks in the dataset.
        threshold: The frequency threshold (e.g., 0.20 for 20%). Keywords
                   appearing in more chunks than this will be removed.
        keep_list: A set of keywords to always keep, regardless of their frequency.

    Returns:
        A new dictionary containing only the filtered, specific keywords.
    """
    threshold = FREQUENCY_THRESHOLD
    keep_list = KEEP_LIST

    filtered_map = {}
    removed_keywords = []

    print(f"\n--- Starting Frequency Filtering ---")
    print(f"Original number of unique keywords: {len(key_chunk_map)}")
    print(f"Total chunks: {total_chunks}")
    print(f"Frequency threshold: {threshold:.0%}")
    print(f"Protected keywords: {keep_list}\n")

    for keyword, chunks in key_chunk_map.items():
        # The number of chunks a keyword is in IS its document frequency
        document_frequency = len(chunks)
        prevalence = document_frequency / total_chunks

        # Keep the keyword if it's in the protected list
        if keyword in keep_list:
            filtered_map[keyword] = chunks
            continue

        # Keep the keyword if its prevalence is below or equal to the threshold
        if prevalence <= threshold:
            filtered_map[keyword] = chunks
        else:
            # Otherwise, it's too generic, so we track it for removal
            removed_keywords.append((keyword, prevalence))

    # --- Reporting ---
    print(f"Filtering complete.")
    print(f"Kept {len(filtered_map)} specific keywords.")
    print(f"Removed {len(removed_keywords)} generic keywords.\n")

    # Sort removed keywords by prevalence to see the most common ones first
    removed_keywords.sort(key=lambda item: item[1], reverse=True)
    
    print("Top 10 most frequent (and removed) keywords:")
    for kw, prevalence in removed_keywords[:10]:
        print(f"- '{kw}' (found in {prevalence:.1%} of chunks)")

    return filtered_map


if __name__ == '__main__':
    # --- Example Usage ---

    # 1. Create some sample data
    chunks_data = [
        "The policy member filed a claim for the benefit.",
        "Insurance benefit claims are processed by the policy administrator.",
        "A new claim was filed by the member for an accidental death benefit.",
        "The principal life insurance company is based in Des Moines.",
        "Principal Financial Services handles the policy.",
        "The member must pay the premium for the insurance policy."
    ]
    TOTAL_CHUNKS_IN_DATASET = len(chunks_data)

    # This map would be the output of your ner_extractor.py
    key_chunk_map = {
        'policy': [chunks_data[0], chunks_data[1], chunks_data[4], chunks_data[5]], # 4/6 = 67%
        'member': [chunks_data[0], chunks_data[2], chunks_data[5]],                # 3/6 = 50%
        'claim': [chunks_data[0], chunks_data[1], chunks_data[2]],                 # 3/6 = 50%
        'benefit': [chunks_data[0], chunks_data[1], chunks_data[2]],               # 3/6 = 50%
        'insurance': [chunks_data[1], chunks_data[3], chunks_data[5]],             # 3/6 = 50%
        'accidental death benefit': [chunks_data[2]],                             # 1/6 = 17%
        'principal life insurance company': [chunks_data[3]],                     # 1/6 = 17%
        'des moines': [chunks_data[3]],                                           # 1/6 = 17%
        'principal financial services': [chunks_data[4]],                         # 1/6 = 17%
        'premium': [chunks_data[5]]                                               # 1/6 = 17%
    }

    # 2. Run the filtering function
    filtered_map = filter_keys(
        key_chunk_map=key_chunk_map,
        total_chunks=TOTAL_CHUNKS_IN_DATASET,
        threshold=FREQUENCY_THRESHOLD,
        keep_list=KEEP_LIST
    )

    # 3. View the results
    print("\n--- Final Filtered Keywords ---")
    for keyword in filtered_map.keys():
        print(f"- {keyword}")