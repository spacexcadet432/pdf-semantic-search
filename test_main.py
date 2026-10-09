import subprocess
import sys
import unittest
from pathlib import Path

import pymupdf

from main import search_pdf


ROOT = Path(__file__).parent
PDF_PATH = ROOT / "book.pdf"


class PdfSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with pymupdf.open(PDF_PATH) as pdf:
            cls.page_count = len(pdf)

    def assert_valid_result(self, result, expected_terms):
        self.assertIsNotNone(result["page_number"])
        self.assertGreaterEqual(result["page_number"], 1)
        self.assertLessEqual(result["page_number"], self.page_count)
        self.assertGreaterEqual(result["similarity"], 0.10)

        text = result["text"].lower()
        for term in expected_terms:
            self.assertIn(term, text)

    def test_finds_python_functions_without_repeated_generic_sentence(self):
        result = search_pdf(PDF_PATH, "How do functions work in Python?")

        self.assert_valid_result(result, ["function", "python"])
        self.assertNotIn(
            "i sleep all night and i work all day",
            result["text"].lower()
        )

    def test_finds_recursion(self):
        result = search_pdf(PDF_PATH, "What is recursion in Python?")

        self.assert_valid_result(result, ["recursion"])

    def test_short_recursion_query_avoids_table_of_contents(self):
        result = search_pdf(PDF_PATH, "What is recursion?")

        self.assert_valid_result(result, ["recursion"])
        self.assertNotIn(". . .", result["text"])

    def test_finds_lists(self):
        result = search_pdf(PDF_PATH, "How do Python lists work?")

        self.assert_valid_result(result, ["list"])

    def test_unrelated_query_has_no_confident_match(self):
        result = search_pdf(PDF_PATH, "How do I bake sourdough bread?")

        self.assertIsNone(result["page_number"])
        self.assertIsNone(result["text"])

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
