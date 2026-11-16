"""Planning-contract benchmarks must use each source query's registered scale."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from bench_certification_methods import limits_of, margin_index_from_records  # noqa: E402
from curobo_certify.margin_campaign import freeze, dump_manifest


class PlanningMarginBenchmarkTests(unittest.TestCase):
    def test_default_and_nondefault_float32_planning_limits(self):
        a={k:np.array([[-10.0,-20.0],[10.0,20.0]],dtype=np.float32)
           for k in ("velocity_limits","acceleration_limits","jerk_limits")}
        default=limits_of(a,"planning")["jerk"]
        self.assertTrue(np.array_equal(default,np.array([9.5,19],dtype=np.float32).astype(float)))
        at975=limits_of(a,"planning",0.975)["jerk"]
        self.assertTrue(np.array_equal(at975,
            (np.array([10,20],dtype=np.float32)*np.float32(.975)).astype(float)))
        self.assertFalse(np.array_equal(default,at975))
        original=limits_of(a,"physical",0.975)["jerk"]
        self.assertTrue(np.array_equal(original,np.array([10.,20.])))
        with self.assertRaises(ValueError):
            limits_of(a,"planning",1.01)

    def test_association_from_frozen_original_npz_sha(self):
        rows=freeze(44,1,["0.95","1.00"])
        source=rows[0]
        with tempfile.TemporaryDirectory() as tmp:
            query=Path(tmp)/"query.json"
            query.write_bytes(dump_manifest(rows))
            records=Path(tmp)/"records";records.mkdir()
            sha="a"*64
            rec={"source_id":source["source_id"],"source_query_sha256":source["source_query_sha256"],
                 "source_query":source,"planner_success":True,"arrays_sha256":sha,
                 "status":"NATIVE_SUCCESS_PARITY_PASS"}
            (records/(source["source_id"]+".json")).write_text(json.dumps(rec))
            idx=margin_index_from_records(query,records)
            self.assertEqual(idx[sha][0],0.95)
            self.assertEqual(idx[sha][1],source["source_id"])

    def test_same_npz_under_different_scale_is_suspicious_and_rejected(self):
        rows=freeze(45,1,["0.95","1.00"])
        with tempfile.TemporaryDirectory() as tmp:
            query=Path(tmp)/"query.json";query.write_bytes(dump_manifest(rows))
            records=Path(tmp)/"records";records.mkdir()
            for row in rows[:2]:
                rec={"source_id":row["source_id"],"source_query_sha256":row["source_query_sha256"],
                     "source_query":row,"planner_success":True,"arrays_sha256":"b"*64,
                     "status":"NATIVE_SUCCESS_PARITY_PASS"}
                (records/(row["source_id"]+".json")).write_text(json.dumps(rec))
            with self.assertRaisesRegex(ValueError,"different margin arms"):
                margin_index_from_records(query,records)


if __name__=="__main__":
    unittest.main()
