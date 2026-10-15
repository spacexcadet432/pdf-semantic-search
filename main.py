
import argparse
import json
import os
import re
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from pathlib import Path

import pymupdf
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


RELEVANCE_THRESHOLD = 0.10
SENTENCE_RELEVANCE_THRESHOLD = 0.08
DEFAULT_LLM_ENDPOINT = "https://api.openai.com/v1/chat/completions"
DEFAULT_LLM_MODEL = "gpt-4o-mini"
MAX_CONTEXT_CHUNKS = 5
EXTRA_STOP_WORDS = {
    "work",
    "works",
    "working",
}
STOP_WORDS = sorted(ENGLISH_STOP_WORDS.union(EXTRA_STOP_WORDS))


def load_dotenv(env_path=None):
    """Load simple KEY=VALUE entries from .env without overriding the shell."""
    env_path = Path(env_path) if env_path else Path(__file__).with_name(".env")
    if not env_path.is_file():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


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


def search_pdf(pdf_path, query, max_chunks=MAX_CONTEXT_CHUNKS):
    """Retrieve the most relevant text chunks from a PDF."""
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

    # Rank passages made from each page. Overlapping windows preserve context
    # while avoiding sending an entire book page to the language model.
    chunks = []
    for page in nonempty_pages:
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", page["text"])
            if sentence.strip()
        ]
        unique_sentences = []
        seen = set()
        for sentence in sentences:
            normalized = normalize_for_search(sentence)
            if normalized not in seen:
                seen.add(normalized)
                unique_sentences.append(sentence)

        if not unique_sentences:
            continue
        sentence_vectors = vectorizer.transform(unique_sentences)
        sentence_scores = cosine_similarity(query_vector, sentence_vectors).flatten()
        # Group neighboring sentences into short coherent context passages.
        for start in range(0, len(unique_sentences), 3):
            passage = unique_sentences[start:start + 3]
            passage_score = float(sentence_scores[start:start + 3].max())
            if passage_score >= SENTENCE_RELEVANCE_THRESHOLD:
                chunks.append({
                    "page_number": page["page_number"],
                    "similarity": passage_score,
                    "text": " ".join(passage),
                })

    chunks.sort(key=lambda chunk: chunk["similarity"], reverse=True)
    selected = chunks[:max_chunks]
    if not selected or selected[0]["similarity"] < RELEVANCE_THRESHOLD:
        return {"chunks": [], "similarity": float(scores.max()) if len(scores) else 0.0}
    return {"chunks": selected, "similarity": selected[0]["similarity"]}


def build_augmented_prompt(query, chunks):
    """Build the grounded prompt sent to the language model."""
    context = "\n\n".join(
        f"[Chunk {index} | PDF page {chunk['page_number']}]\n{chunk['text']}"
        for index, chunk in enumerate(chunks, start=1)
    )
    return (
        "Answer the user's question using the context below. If the context "
        "does not contain the answer, say that you could not find the answer "
        "in the PDF. Do not invent facts. Cite supporting pages as [page N].\n\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    )


def generate_response(prompt, api_key=None, endpoint=None, model=None, timeout=60):
    """Call an OpenAI-compatible chat completions endpoint."""
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    endpoint = endpoint or os.environ.get("OPENAI_BASE_URL", DEFAULT_LLM_ENDPOINT)
    model = model or os.environ.get("OPENAI_MODEL", DEFAULT_LLM_MODEL)
    if not api_key:
        raise ValueError(
            "Missing LLM API key. Set OPENAI_API_KEY to use generation."
        )

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }).encode("utf-8")
    request = Request(
        endpoint,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM request failed ({error.code}): {details}") from error
    except (URLError, TimeoutError) as error:
        raise RuntimeError(f"Could not reach the LLM endpoint: {error}") from error
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("LLM endpoint returned an unexpected response.") from error


def main():
    load_dotenv()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    parser = argparse.ArgumentParser(
        description="Ask questions about a PDF using retrieval-augmented generation."
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

        print("\n========== PDF RAG RESPONSE ==========")
        print(f"Query: {query}")
        if not result["chunks"]:
            print(
                "\nNo relevant context found in the PDF, so no LLM request was made."
            )
        else:
            prompt = build_augmented_prompt(query, result["chunks"])
            print("\n========== AUGMENTED PROMPT ==========")
            print(prompt)
            print("\n========== GENERATED RESPONSE ==========")
            print(generate_response(prompt))

        print("=======================================")

    except (ValueError, RuntimeError) as error:
        print(f"Error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
