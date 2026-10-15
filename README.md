# PDF RAG CLI

This command-line tool retrieves relevant passages from a PDF, adds them to a
prompt with your question, and asks an OpenAI-compatible chat-completions API
to generate a grounded answer. It prints both the augmented prompt and answer.

## Setup

Use Python 3.10 or newer. Install the required packages:

```powershell
python -m pip install pymupdf scikit-learn
```

Copy `.env.example` to `.env` in the project folder and set your credentials:

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1/chat/completions
OPENAI_MODEL=gpt-4o-mini
```

The CLI loads `.env` automatically. Values already set in the shell take
precedence over values in `.env`. Keep `.env` private; it is excluded by
`.gitignore`.

For a compatible provider, set `OPENAI_BASE_URL` to its chat-completions URL
and `OPENAI_MODEL` to a model available to your account. The key is sent as a
Bearer token.

## Run

```powershell
python main.py book.pdf "What is recursion in Python?"
```

The tool selects up to five relevant passages. If no relevant context is
found, it reports that and does not call the LLM. Otherwise, it displays the
augmented prompt and generated response.
