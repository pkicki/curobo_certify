import tempfile
import unittest
from pathlib import Path

from scripts.export_frozen_pilot_models import metadata_paths


class MetadataSelectionTests(unittest.TestCase):
    def test_keeps_panda_query_families_and_excludes_summary_and_other_robots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("pilot-panda-00.json", "explore-panda-01.json", "summary.json",
                         "pilot-ur5e-00.json", "notes.json"):
                (root / name).write_text("{}")
            self.assertEqual([p.name for p in metadata_paths(root)],
                             ["explore-panda-01.json", "pilot-panda-00.json"])


if __name__ == "__main__":
    unittest.main()
