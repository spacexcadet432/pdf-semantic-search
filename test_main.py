import subprocess
import sys
import os
import unittest
from pathlib import Path

import pymupdf

from main import search_pdf
from main import build_augmented_prompt, load_dotenv


ROOT = Path(__file__).parent
PDF_PATH = ROOT / "book.pdf"


class PdfSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with pymupdf.open(PDF_PATH) as pdf:
            cls.page_count = len(pdf)

    def assert_valid_result(self, result, expected_terms):
        self.assertTrue(result["chunks"])
        self.assertGreaterEqual(result["similarity"], 0.10)

        text = " ".join(chunk["text"] for chunk in result["chunks"]).lower()
        for term in expected_terms:
            self.assertIn(term, text)

    def test_finds_python_functions_without_repeated_generic_sentence(self):
        result = search_pdf(PDF_PATH, "How do functions work in Python?")

        self.assert_valid_result(result, ["function", "python"])
        self.assertNotIn(
            "i sleep all night and i work all day",
            " ".join(chunk["text"] for chunk in result["chunks"]).lower()
        )

    def test_finds_recursion(self):
        result = search_pdf(PDF_PATH, "What is recursion in Python?")

        self.assert_valid_result(result, ["recursion"])

    def test_short_recursion_query_avoids_table_of_contents(self):
        result = search_pdf(PDF_PATH, "What is recursion?")

        self.assert_valid_result(result, ["recursion"])
        self.assertNotIn(
            ". . .",
            " ".join(chunk["text"] for chunk in result["chunks"]),
        )

    def test_finds_lists(self):
        result = search_pdf(PDF_PATH, "How do Python lists work?")

        self.assert_valid_result(result, ["list"])

    def test_unrelated_query_has_no_confident_match(self):
        result = search_pdf(PDF_PATH, "How do I bake sourdough bread?")

        self.assertEqual(result["chunks"], [])

    def test_augmented_prompt_contains_question_context_and_citations(self):
        prompt = build_augmented_prompt("What is recursion?", [{
            "page_number": 12,
            "text": "A function can call itself.",
        }])
        self.assertIn("What is recursion?", prompt)
        self.assertIn("PDF page 12", prompt)
        self.assertIn("A function can call itself.", prompt)
        self.assertIn("say that you could not find", prompt)

    def test_load_dotenv_reads_values_without_overriding_existing_environment(self):
        env_path = ROOT / ".test-rag.env"
        env_path.write_text(
            '# configuration\nTEST_RAG_ENV="loaded value"\n'
            'TEST_RAG_EXISTING=from-file\nINVALID_LINE\n',
            encoding="utf-8",
        )
        os.environ.pop("TEST_RAG_ENV", None)
        os.environ["TEST_RAG_EXISTING"] = "from-shell"
        try:
            load_dotenv(env_path)
            self.assertEqual(os.environ["TEST_RAG_ENV"], "loaded value")
            self.assertEqual(os.environ["TEST_RAG_EXISTING"], "from-shell")
        finally:
            env_path.unlink(missing_ok=True)
            os.environ.pop("TEST_RAG_ENV", None)
            os.environ.pop("TEST_RAG_EXISTING", None)

    def test_cli_reports_missing_file(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "main.py"),
                str(ROOT / "missing.pdf"),
                "functions",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("Error: File not found", completed.stdout)

    def test_cli_rejects_non_pdf_input(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "main.py"),
                str(ROOT / "main.py"),
                "functions",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("Error: The input file must be a PDF.", completed.stdout)


if __name__ == "__main__":
    unittest.main()
