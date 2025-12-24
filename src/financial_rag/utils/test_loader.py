"""Test data loader for gold Q&A evaluation data.

Loads and filters test cases from the gold Q&A CSV file based on configuration.
Converts test cases to promptfoo-compatible format.
"""

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from financial_rag.config import get_app_config, get_settings


@dataclass
class TestCase:
    """A single test case from the gold Q&A dataset."""

    question: str
    source_docs: str
    question_type: str
    source_chunk_type: str
    answer: str

    def to_promptfoo_format(self) -> dict:
        """Convert to promptfoo test case format.

        Returns:
            Dictionary in promptfoo test case format.
        """
        return {
            "vars": {
                "query": self.question,
                "source_docs": self.source_docs,
                "question_type": self.question_type,
                "chunk_type": self.source_chunk_type,
                "ground_truth": self.answer,
            },
            "assert": [
                {
                    "type": "factuality",
                    "value": "{{ground_truth}}",
                },
                {
                    "type": "llm-rubric",
                    "value": (
                        "Response accurately addresses the question "
                        "using data from the specified source documents."
                    ),
                },
            ],
        }


def parse_source_docs_pattern(source_docs: str) -> list[str]:
    """Parse source docs pattern to extract ticker symbols.

    Examples:
        "*AAPL*" -> ["AAPL"]
        "*2023 Q3 AAPL*" -> ["2023 Q3 AAPL"]
        "2023 Q3 AAPL.pdf, 2022 Q3 AAPL.pdf" -> ["2023 Q3 AAPL", "2022 Q3 AAPL"]

    Args:
        source_docs: Source docs pattern from CSV.

    Returns:
        List of document identifiers (without .pdf extension).
    """
    # Remove asterisks and clean up
    cleaned = source_docs.replace("*", "").strip()

    # Check if it's a list of documents
    if "," in cleaned:
        docs = [d.strip().replace(".pdf", "") for d in cleaned.split(",")]
        return docs

    # Single document or ticker pattern
    return [cleaned.replace(".pdf", "")]


def matches_selected_documents(source_docs: str, selected_documents: list[str]) -> bool:
    """Check if source docs pattern matches any selected document.

    Args:
        source_docs: Source docs pattern from CSV (e.g., "*AAPL*", "*2023 Q3 AAPL*").
        selected_documents: List of selected document filenames.

    Returns:
        True if pattern matches at least one selected document.
    """
    # Parse the pattern to get identifiers
    identifiers = parse_source_docs_pattern(source_docs)

    # Check each identifier against selected documents
    for identifier in identifiers:
        for doc in selected_documents:
            # Remove .pdf for comparison
            doc_name = doc.replace(".pdf", "")

            # Check if identifier is a ticker (e.g., "AAPL") or full name (e.g., "2023 Q3 AAPL")
            if identifier.upper() in doc_name.upper():
                return True

    return False


def get_matching_documents(source_docs: str, selected_documents: list[str]) -> list[str]:
    """Get list of selected documents that match the source docs pattern.

    Args:
        source_docs: Source docs pattern from CSV.
        selected_documents: List of selected document filenames.

    Returns:
        List of matching document filenames.
    """
    identifiers = parse_source_docs_pattern(source_docs)
    matching = []

    for identifier in identifiers:
        for doc in selected_documents:
            doc_name = doc.replace(".pdf", "")
            if identifier.upper() in doc_name.upper() and doc not in matching:
                matching.append(doc)

    return matching


class TestLoader:
    """Load and filter test cases from gold Q&A CSV."""

    def __init__(self, csv_path: Path | None = None):
        """Initialize test loader.

        Args:
            csv_path: Path to gold Q&A CSV file. Defaults to config path.
        """
        settings = get_settings()
        self._app_config = get_app_config()

        self._csv_path = csv_path or settings.gold_qn_dir / "qna_data.csv"
        self._test_cases: list[TestCase] = []
        self._loaded = False

    @property
    def question_types(self) -> list[str]:
        """Get configured question types to include."""
        return self._app_config.question_types

    @property
    def chunk_types(self) -> list[str]:
        """Get configured chunk types to include."""
        return self._app_config.chunk_types

    @property
    def selected_documents(self) -> list[str]:
        """Get configured selected documents."""
        return self._app_config.selected_documents

    @property
    def max_test_cases(self) -> int | None:
        """Get maximum number of test cases to load."""
        return self._app_config.max_test_cases

    def _load_csv(self) -> list[TestCase]:
        """Load all test cases from CSV.

        Returns:
            List of all test cases.
        """
        test_cases = []

        with open(self._csv_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            for row in reader:
                test_case = TestCase(
                    question=row["Question"],
                    source_docs=row["Source Docs"],
                    question_type=row["Question Type"],
                    source_chunk_type=row["Source Chunk Type"],
                    answer=row["Answer"],
                )
                test_cases.append(test_case)

        return test_cases

    def load(self) -> list[TestCase]:
        """Load and filter test cases based on configuration.

        Filters by:
        - Question types (from config)
        - Chunk types (from config)
        - Selected documents (from config)
        - Max test cases limit (from config)

        Returns:
            Filtered list of test cases.
        """
        all_cases = self._load_csv()
        filtered = []

        for case in all_cases:
            # Filter by question type
            if self.question_types and case.question_type not in self.question_types:
                continue

            # Filter by chunk type
            if self.chunk_types and case.source_chunk_type not in self.chunk_types:
                continue

            # Filter by selected documents
            if self.selected_documents:
                if not matches_selected_documents(case.source_docs, self.selected_documents):
                    continue

            filtered.append(case)

        # Apply max test cases limit
        if self.max_test_cases and len(filtered) > self.max_test_cases:
            filtered = filtered[: self.max_test_cases]

        self._test_cases = filtered
        self._loaded = True

        return filtered

    def get_test_cases(self) -> list[TestCase]:
        """Get loaded test cases, loading if necessary.

        Returns:
            List of filtered test cases.
        """
        if not self._loaded:
            self.load()
        return self._test_cases

    def to_promptfoo_tests(self) -> list[dict]:
        """Convert test cases to promptfoo format.

        Returns:
            List of test cases in promptfoo format.
        """
        return [case.to_promptfoo_format() for case in self.get_test_cases()]

    def export_promptfoo_yaml(self, output_path: Path) -> None:
        """Export test cases to a promptfoo YAML file.

        Args:
            output_path: Path to output YAML file.
        """
        tests = self.to_promptfoo_tests()

        output_data = {"tests": tests}

        with open(output_path, "w") as f:
            yaml.dump(output_data, f, default_flow_style=False, allow_unicode=True)

    def get_statistics(self) -> dict:
        """Get statistics about loaded test cases.

        Returns:
            Dictionary with test case statistics.
        """
        cases = self.get_test_cases()

        # Count by question type
        by_question_type = {}
        for case in cases:
            by_question_type[case.question_type] = by_question_type.get(case.question_type, 0) + 1

        # Count by chunk type
        by_chunk_type = {}
        for case in cases:
            by_chunk_type[case.source_chunk_type] = by_chunk_type.get(case.source_chunk_type, 0) + 1

        # Count by ticker (extract from source docs)
        by_ticker = {}
        for case in cases:
            matching = get_matching_documents(case.source_docs, self.selected_documents)
            for doc in matching:
                # Extract ticker from document name
                match = re.search(r"([A-Z]{2,5})\.pdf", doc)
                if match:
                    ticker = match.group(1)
                    by_ticker[ticker] = by_ticker.get(ticker, 0) + 1

        return {
            "total_test_cases": len(cases),
            "by_question_type": by_question_type,
            "by_chunk_type": by_chunk_type,
            "by_ticker": by_ticker,
            "filters_applied": {
                "question_types": self.question_types,
                "chunk_types": self.chunk_types,
                "selected_documents": self.selected_documents,
                "max_test_cases": self.max_test_cases,
            },
        }
