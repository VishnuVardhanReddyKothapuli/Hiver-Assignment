"""Real-data integration checks; skip only when the snapshot is not present."""
import unittest
from support_lab.common import DATA, ARTIFACTS, read_csv, read_jsonl
from support_lab.evaluation import audit_splits

@unittest.skipUnless((DATA/"manifest.json").exists(), "Frozen dataset absent")
class DataIntegrityTests(unittest.TestCase):
    def test_frozen_dataset_disjointness(self):
        parts = {s: read_jsonl(DATA/f"{s}.jsonl") for s in ("train", "dev", "test")}
        checks = audit_splits(parts)
        self.assertTrue(all(n == 0 for n in checks.values()))
        self.assertEqual(len(parts["dev"]), 40)
        self.assertEqual(len(parts["test"]), 160)
        for name in ("dev", "test"):
            self.assertEqual(len({r["group_id"] for r in parts[name]}), len(parts[name]))

    def test_reply_provenance_is_train_only(self):
        path = ARTIFACTS/"diagnostic_predictions.jsonl"
        if not path.exists():
            self.skipTest("Run diagnostics first")
        train = {r["reply_id"]: r for r in read_jsonl(DATA/"train.jsonl")}
        test_ids = {r["id"] for r in read_jsonl(DATA/"test.jsonl")}
        predictions = read_jsonl(path)
        self.assertEqual(len(predictions), 480)
        self.assertEqual(len({(r["id"],r["system"]) for r in predictions}),480)
        for p in predictions:
            self.assertIn(p["id"], test_ids)
            for evidence in p["evidence"]:
                self.assertIn(evidence["reply_id"], train)
                self.assertEqual(evidence["reply"], train[evidence["reply_id"]]["reply"])

    def test_blinded_reply_pool_is_paired_and_partitioned_by_message(self):
        if not (DATA/"reply_review.jsonl").exists():
            self.skipTest("Reply pool not prepared")
        rows = read_jsonl(DATA/"reply_review.jsonl")
        self.assertEqual(len(rows),60)
        self.assertTrue(all("system" not in r for r in rows))
        a = {r["message_id"] for r in rows if r["partition"] == "calibration"}
        b = {r["message_id"] for r in rows if r["partition"] == "audit"}
        self.assertEqual((len(a),len(b)),(10,10))
        self.assertFalse(a & b)
        for tid in a | b:
            self.assertEqual(sum(r["message_id"] == tid for r in rows),3)

if __name__ == "__main__":
    unittest.main()
