from collections import Counter

# doc_id_list = [5, 6, 16, 17, 18, 21, 30, 32, 33, 39, 45, 47, 56, 64, 71, 81, 85, 85, 85, 6, 506]

def topk_most_freq_id(doc_id_list, k):
    # Count frequencies
    counter = Counter(doc_id_list)

    # Get the most common elements (top k)
    k_most_common = counter.most_common(k)

    # Extract only the elements (you can also keep counts if needed)
    top_elements = [item[0] for item in k_most_common]

    return top_elements
