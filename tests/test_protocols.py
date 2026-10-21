import tempfile
import unittest
from pathlib import Path

from wifi_csi_quality.protocols import COMMON_LABELS, scan_xrf55_common


class ProtocolTests(unittest.TestCase):
    def test_xrf55_common_label_mapping(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            for action_id in (31, 34, 35, 39, 1):
                path = root / "Scene_1" / "lb" / "01" / f"01_{action_id:02d}_01.dat"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            records = scan_xrf55_common(root)
        self.assertEqual(len(records), 4)
        self.assertEqual({record["label"] for record in records}, set(COMMON_LABELS))
        self.assertEqual(len({record["sample_id"] for record in records}), 4)

    def test_rejects_subject_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            path = root / "Scene_1" / "lb" / "02" / "01_31_01.dat"
            path.parent.mkdir(parents=True)
            path.touch()
            with self.assertRaises(ValueError):
                scan_xrf55_common(root)


if __name__ == "__main__":
    unittest.main()

