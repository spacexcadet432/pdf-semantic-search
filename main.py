
import argparse
import re
import sys
from pathlib import Path

import pymupdf
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


RELEVANCE_THRESHOLD = 0.10
SENTENCE_RELEVANCE_THRESHOLD = 0.08
EXTRA_STOP_WORDS = {
    "work",
    "works",
    "working",
}
STOP_WORDS = sorted(ENGLISH_STOP_WORDS.union(EXTRA_STOP_WORDS))


def is_reference_page(text):
    """Skip table-of-contents style pages for normal content searches."""
    lowered = text.lower()
    first_words = lowered[:80]

    return (
        first_words.startswith("contents")
        or re.search(r"\bcontents\b", first_words)
        and lowered.count(". . .") >= 3
    )


def normalize_text(text):
    """Normalize PDF text artifacts that hurt matching and console output."""
    replacements = {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return re.sub(r"\s+", " ", text).strip()


def normalize_for_search(text):
    """Apply small query/document normalizations without changing output text."""
    text = normalize_text(text).lower()
    # Smooth common singular/plural mismatches in chapter-topic queries.
    text = re.sub(r"\b(function|list)s\b", r"\1", text)
    return text


def extract_pages(pdf_path):
    """Extract text from each PDF page."""
    pages = []

    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text")
            text = normalize_text(text)

            pages.append({
                "page_number": page_number,
                "text": text
            })

    return pages


def search_pdf(pdf_path, query):
    """Find the page most similar to the query."""
    pages = extract_pages(pdf_path)

    # Ignore blank and table-of-contents style pages.
    nonempty_pages = [
        p for p in pages
        if p["text"] and not is_reference_page(p["text"])
    ]

    if not nonempty_pages:
        raise ValueError("No extractable text found in the PDF.")

    page_texts = [normalize_for_search(p["text"]) for p in nonempty_pages]

    # Convert pages into TF-IDF vectors.
    vectorizer = TfidfVectorizer(
        preprocessor=normalize_for_search,
        stop_words=STOP_WORDS,
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_df=0.85
    )

    page_vectors = vectorizer.fit_transform(page_texts)

    # Convert the query using the same vocabulary.
    query_vector = vectorizer.transform([query])

    # Compare the query against every page.
    scores = cosine_similarity(
        query_vector, page_vectors
    ).flatten()

    best_index = scores.argmax()
    best_page = nonempty_pages[best_index]
    best_score = float(scores[best_index])

    if best_score < RELEVANCE_THRESHOLD:
        return {
            "page_number": None,
            "similarity": best_score,
            "text": None
        }

    # Find relevant sentences on the best page.
    sentences = re.split(
        r"(?<=[.!?])\s+",
        best_page["text"]
    )
    sentences = [s.strip() for s in sentences if s.strip()]

    relevant_text = best_page["text"]

    if sentences:
        unique_sentences = []
        seen = set()

        for sentence in sentences:
            normalized = normalize_for_search(sentence)

            if normalized in seen:
                continue

            seen.add(normalized)
            unique_sentences.append(sentence)

        sentence_vectors = vectorizer.transform(unique_sentences)
        sentence_scores = cosine_similarity(
            query_vector, sentence_vectors
        ).flatten()

        ranked_indices = sentence_scores.argsort()[::-1]
        selected = []

        for index in ranked_indices:
            if sentence_scores[index] < SENTENCE_RELEVANCE_THRESHOLD:
                break

            selected.append(unique_sentences[index])

            if len(selected) == 3:
                break

        if selected:
            relevant_text = " ".join(selected)

    return {
        "page_number": best_page["page_number"],
        "similarity": best_score,
        "text": relevant_text
    }


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    parser = argparse.ArgumentParser(
        description="Find the most relevant page in a PDF."
    )

    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument(
        "query",
        nargs="+",
        help="Question or search query"
    )

    args = parser.parse_args()
    pdf_path = Path(args.pdf)
    query = " ".join(args.query).strip()

    if not pdf_path.is_file():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    if pdf_path.suffix.lower() != ".pdf":
        print("Error: The input file must be a PDF.")
        sys.exit(1)

    if not query:
        print("Error: Query cannot be empty.")
        sys.exit(1)

    try:
        result = search_pdf(pdf_path, query)

        print("\n========== PDF SEARCH RESULT ==========")
        print(f"Query: {query}")
        print(f"Similarity score: {result['similarity']:.4f}")

        if result["page_number"] is None:
            print(
                "\nNo confident match found. Try a more specific query or "
                "check whether the PDF covers this topic."
            )
        else:
            print(f"Page number: {result['page_number']}")
            print("\nRelevant text:")
            print(result["text"])

        print("=======================================")

    except (ValueError, RuntimeError) as error:
        print(f"Error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
