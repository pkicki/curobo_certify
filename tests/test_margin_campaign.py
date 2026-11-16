"""Offline regression gates for matched planning-margin experiment integrity."""
import json
import math
import tempfile
import unittest
from pathlib import Path

from curobo_certify.margin_campaign import (
    as_margin, margin_label, parse_label, freeze, check_manifest, dump_manifest, digest)
from curobo_certify.margin_analysis import build_analysis, paired_bootstrap


class MarginCampaignTests(unittest.TestCase):
    def setUp(self):
        self.rows=freeze(1234,1,["0.95","1.00"])

    def test_frozen_pairs_and_sha_are_reproducible(self):
        self.assertEqual(len(self.rows),24)
        self.assertEqual(len(dump_manifest(self.rows)),len(dump_manifest(freeze(1234,1,["0.95","1.00"]))))
        self.assertEqual(digest(dump_manifest(self.rows)),
                         digest(dump_manifest(freeze(1234,1,["0.95","1.00"]))))
        self.assertEqual(check_manifest(self.rows)["pairs"],12)
        pair=self.rows[:2]
        self.assertEqual(pair[0]["goal_joint_positions_rad"],pair[1]["goal_joint_positions_rad"])
        self.assertNotEqual(pair[0]["source_id"],pair[1]["source_id"])

    def test_margin_labels_reversible(self):
        for level in ("0.9","0.95","0.975","0.99","1.00"):
            self.assertEqual(parse_label(margin_label(level)),as_margin(level))
        self.assertEqual(parse_label("default_margin_0p95"),as_margin("0.95"))
        self.assertEqual(parse_label("margin_arm_1p00"),as_margin("1.00"))

    def test_invalid_and_duplicate_scales_fail_closed(self):
        for scale in ("-0.1","0","1.01","NaN","Inf","2e-1","0.1234"):
            with self.assertRaises(ValueError):
                freeze(1,1,[scale])
        with self.assertRaises(ValueError):
            freeze(1,1,["0.95","0.950"])
        with self.assertRaises(ValueError):
            freeze(1,0,["0.95"])

    def test_tampered_query_or_incomplete_pair_is_rejected(self):
        row=dict(self.rows[0])
        row["goal_delta_rad"]=[0.0]*7
        with self.assertRaises(ValueError):
            check_manifest([row]+self.rows[1:])
        with self.assertRaises(ValueError):
            check_manifest(self.rows[:-1])
        with self.assertRaises(ValueError):
            check_manifest(self.rows+self.rows[:1])

    @staticmethod
    def native_record(query,duration,sha):
        return {"source_id":query["source_id"],"source_query_sha256":query["source_query_sha256"],
                "source_query":query,"planner_success":True,"native_duration_seconds":duration,
                "arrays_sha256":sha,"status":"NATIVE_SUCCESS_PARITY_PASS",
                "solver_condition":query["solver_condition"]}

    def _fixture(self,folder):
        path=Path(folder)
        record_dir=path/"records";record_dir.mkdir()
        left,right=self.rows[:2]
        for row,duration,sha in ((left,1.0,"a"*64),(right,0.9,"b"*64)):
            (record_dir/(row["source_id"]+".json")).write_text(
                json.dumps(self.native_record(row,duration,sha)))
        failure=self.rows[2]
        (record_dir/(failure["source_id"]+".json")).write_text(json.dumps(
            {"source_id":failure["source_id"],"source_query_sha256":failure["source_query_sha256"],
             "status":"FAILED_RETAINED"}))
        return record_dir,left,right

    def test_missing_and_failed_outputs_retained_in_denominator(self):
        with tempfile.TemporaryDirectory() as t:
            records,left,right=self._fixture(t)
            a=build_analysis(self.rows,records,bootstrap=100)
            self.assertEqual(a["denominators"]["source_queries"],24)
            self.assertEqual(a["denominators"]["attempted_gpu_queries"],3)
            self.assertEqual(a["summary_by_margin"]["0.95"]["native_success"],1)
            self.assertEqual(a["summary_by_margin"]["1.00"]["native_success"],1)
            self.assertEqual(a["summary_by_margin"]["0.95"]["native_failed"],1)
            self.assertEqual(a["summary_by_margin"]["0.95"]["certification_not_run"],12)
            pair=a["paired_to_0p95"]["1.00"]["duration_gain_pct_paired_success_only"]
            self.assertEqual(pair["n"],1)
            self.assertAlmostEqual(pair["median"],10)
            self.assertEqual(pair["ci95"],[10,10])
            self.assertEqual(a["summary_by_margin"]["1.00"]["full_physical_time_certified"],0)

    def test_correct_physical_vs_native_certification_status(self):
        with tempfile.TemporaryDirectory() as t:
            records,left,right=self._fixture(t)
            benchmark=Path(t)/"certs.json"
            benchmark.write_text(json.dumps({"scope":"FROZEN_OUTPUT_FULL_CERTIFICATE_CPU_EXPLORATORY",
                "rows":[
                    {"method":"A","contract":"physical","source_npz_sha256":"a"*64,
                     "final":"CERTIFIED","repeats_ms_total":[2.8,3.0]},
                    {"method":"A","contract":"native","source_npz_sha256":"a"*64,
                     "final":"CERTIFIED","repeats_ms_total":[12.8,13.0]},
                    {"method":"A","contract":"physical","source_npz_sha256":"b"*64,
                     "final":"VIOLATED","repeats_ms_total":[2.8]},
                    {"method":"A","contract":"native","source_npz_sha256":"b"*64,
                     "final":"UNKNOWN_TIGHTENED_WITNESS","repeats_ms_total":[13.0]}
                ]}))
            analysis=build_analysis(self.rows,records,benchmark_file=benchmark,bootstrap=100)
            self.assertEqual(analysis["summary_by_margin"]["0.95"]["model_certified"],1)
            self.assertEqual(analysis["summary_by_margin"]["0.95"]["native_reported_certified"],1)
            self.assertEqual(analysis["summary_by_margin"]["1.00"]["model_violated"],1)
            self.assertEqual(analysis["summary_by_margin"]["1.00"]["native_reported_unknown_tightened"],1)
            self.assertEqual(analysis["summary_by_margin"]["0.95"]["full_physical_time_certified"],0)

    def test_tightened_violation_must_not_be_native_violated(self):
        with tempfile.TemporaryDirectory() as t:
            records,_,_=self._fixture(t)
            benchmark=Path(t)/"certs.json"
            benchmark.write_text(json.dumps({"scope":"FROZEN_OUTPUT_FULL_CERTIFICATE_CPU_EXPLORATORY",
                "rows":[{"method":"A","contract":"native","source_npz_sha256":"a"*64,
                         "final":"VIOLATED","repeats_ms_total":[1]}]}))
            with self.assertRaisesRegex(ValueError,"tightened-limit"):
                build_analysis(self.rows,records,benchmark_file=benchmark,bootstrap=100)

    def test_query_mismatch_and_unregistered_results_fail_closed(self):
        with tempfile.TemporaryDirectory() as t:
            records,left,_=self._fixture(t)
            path=records/(left["source_id"]+".json")
            content=json.loads(path.read_text());content["source_query_sha256"]="f"*64
            path.write_text(json.dumps(content))
            with self.assertRaises(ValueError):
                build_analysis(self.rows,records,bootstrap=100)
            path.write_text(json.dumps(self.native_record(left,1.0,"a"*64)))
            (records/"alien.json").write_text("{}")
            with self.assertRaisesRegex(ValueError,"unregistered"):
                build_analysis(self.rows,records,bootstrap=100)

    def test_invalid_or_missing_npz_when_verification_requested(self):
        with tempfile.TemporaryDirectory() as t:
            records,_,_=self._fixture(t)
            with self.assertRaisesRegex(ValueError,"missing frozen"):
                build_analysis(self.rows,records,verify_arrays=True,bootstrap=100)

    def test_bootstrap_rejects_nonfinite(self):
        with self.assertRaises(ValueError):
            paired_bootstrap([math.nan],samples=100)


if __name__=="__main__":
    unittest.main()
