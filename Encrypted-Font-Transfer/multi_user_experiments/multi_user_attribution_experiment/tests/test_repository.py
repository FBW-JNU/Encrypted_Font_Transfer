import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryTest(unittest.TestCase):
    def test_public_files_do_not_use_generation_labels(self):
        labels = [prefix + suffix for prefix, suffix in (("r", "2"), ("v", "2"), ("r", "8"))]
        forbidden = re.compile(r"(?i)(?<![a-z0-9])(?:" + "|".join(labels) + r")(?![a-z0-9])")
        inspected = []
        for path in ROOT.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".md", ".json", ".txt"}:
                inspected.append(path)
                self.assertIsNone(forbidden.search(path.read_text(encoding="utf-8")), str(path))
        self.assertTrue(inspected)

    def test_training_code_has_no_print_calls(self):
        source = (ROOT / "src" / "trainer.py").read_text(encoding="utf-8")
        self.assertNotRegex(source, r"\bprint\s*\(")
        self.assertIn("tqdm(", source)


if __name__ == "__main__":
    unittest.main()
