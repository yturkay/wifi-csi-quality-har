import json
import tempfile
import unittest
from pathlib import Path

from wifi_csi_quality.inventory import inventory_xrf55, render_markdown


class InventoryTests(unittest.TestCase):
    def test_xrf55_inventory_counts_valid_paths(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            sample = root / "Scene_1" / "lb" / "01" / "01_01_01.dat"
            sample.parent.mkdir(parents=True)
            sample.write_bytes(b"1234")

            result = inventory_xrf55(root)

        self.assertEqual(result["file_count"], 1)
        self.assertEqual(result["size_bytes"], 4)
        self.assertEqual(result["files_by_scene"], {"Scene_1": 1})
        self.assertEqual(result["files_by_receiver"], {"lb": 1})
        self.assertEqual(result["files_by_extension"], {"dat": 1})
        self.assertEqual(result["subjects_by_scene"], {"Scene_1": ["01"]})
        self.assertEqual(result["malformed_path_count"], 0)

    def test_markdown_is_serializable_summary(self) -> None:
        payload = {
            "generated_at_utc": "2026-01-01T00:00:00+00:00",
            "datasets": {
                "csi_bench": {
                    "sample_metadata_rows": 1,
                    "missing_sample_paths": 0,
                    "distributions": {
                        "activity": {"walking": 1},
                        "user": {"U01": 1},
                        "environment": {"E01": 1},
                        "device": {"D01": 1},
                    },
                    "split_counts": {"train_id": 1},
                },
                "xrf55": {
                    "file_count": 1,
                    "size_gib": 0.001,
                    "files_by_scene": {"Scene_1": 1},
                    "files_by_receiver": {"lb": 1},
                    "files_by_extension": {"dat": 1},
                    "malformed_path_count": 0,
                },
            },
        }

        text = render_markdown(json.loads(json.dumps(payload)))
        self.assertIn("CSI-Bench HAR", text)
        self.assertIn("XRF55 Wi-Fi", text)


if __name__ == "__main__":
    unittest.main()

