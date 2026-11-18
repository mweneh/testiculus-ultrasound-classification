"""Report-only checks: saved numeric records, manuscript tables and identity guards."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_evidence import OUT, collect, identities, numeric_metrics, render_tables


class ReportEvidenceTests(unittest.TestCase):
    def test_local_artifacts_match_report_snapshot(self):
        self.assertEqual(collect(), json.loads((OUT / 'evidence.json').read_text()))

    def test_generated_tables_are_current(self):
        data = json.loads((OUT / 'evidence.json').read_text())
        for name, content in render_tables(data).items():
            self.assertEqual(content, (OUT / 'generated' / name).read_text())

    def test_duplicate_paths_are_rejected(self):
        with self.assertRaises(AssertionError):
            identities([{'image_path': 'a.png', 'label': '0'},
                        {'image_path': 'a.png', 'label': '1'}])

    def test_threshold_boundary_and_majority_baseline(self):
        rows = [{'label': '0', 'probability_class_1': '0.1'},
                {'label': '0', 'probability_class_1': '0.2'},
                {'label': '1', 'probability_class_1': '0.5'}]
        boundary = numeric_metrics(rows, .5)
        self.assertEqual(boundary['confusion_matrix'], [[2, 0], [0, 1]])
        baseline = numeric_metrics(rows, 1.)
        self.assertEqual(baseline['accuracy'], 2 / 3)
        self.assertEqual(baseline['balanced_accuracy'], .5)
        self.assertEqual(baseline['recall_sensitivity'], 0.)
        self.assertEqual(baseline['f1'], 0.)


if __name__ == '__main__':
    unittest.main()
