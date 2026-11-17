import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from curobo_certify import formal_cohorts as FC  # noqa: E402
from curobo_certify.margin_campaign import check_manifest  # noqa: E402


class FormalCohortTests(unittest.TestCase):
    def test_deterministic_and_paired(self):
        a, b = FC.freeze_cohort("Panda-Dev40"), FC.freeze_cohort("Panda-Dev40")
        self.assertEqual(FC.dump(a), FC.dump(b))
        info = FC.check_cohort(a, "Panda-Dev40")
        self.assertEqual(info["pairs"], 70)
        self.assertEqual(sum(info["strata_counts_in_pool"].values()), 70)
        self.assertLessEqual(max(info["strata_counts_in_pool"].values()) - min(info["strata_counts_in_pool"].values()), 1)
        check_manifest(a)  # compatible with the frozen benchmark registry

    def test_seeds_distinct_small_and_unused(self):
        seeds = {FC.cohort_seed(c) for c in FC.COHORTS}
        self.assertEqual(len(seeds), 3)
        self.assertTrue(all(0 < s < 2**31 for s in seeds))
        self.assertFalse(seeds & {20261110, 20261111, 20261116, 20261117})

    def test_tamper_rejected(self):
        rows = FC.freeze_cohort("Panda-Dev40")
        bad = copy.deepcopy(rows)
        bad[3]["goal_joint_positions_rad"][0] += 1e-3
        with self.assertRaises(ValueError):
            FC.check_cohort(bad, "Panda-Dev40")
        with self.assertRaises(ValueError):
            FC.check_cohort(rows[:-2], "Panda-Dev40")

    def test_leakage_detects_reused_goal_and_seed(self):
        dev = FC.freeze_cohort("Panda-Dev40")
        prior = {"x": [dict(dev[0], seed=dev[0]["seed"], source_id="other")]}
        rep = FC.leakage_audit({"Panda-Dev40": dev}, prior)
        self.assertFalse(rep["pass"])
        rep = FC.leakage_audit({"Panda-Dev40": dev}, {"x": [dict(dev[0], seed=1, source_id="o",
                                                                   goal_joint_positions_rad=[9.0] * 7)]})
        self.assertTrue(rep["pass"])

    def test_first_k(self):
        st = [(0, True), (1, False), (2, True), (3, True), (4, True)]
        sel = FC.first_k_selection(st, 3)
        self.assertEqual(sel, {"attempted": [0, 1, 2, 3], "selected": [0, 2, 3], "shortfall": 0})
        self.assertEqual(FC.first_k_selection([(0, False), (1, True)], 3)["shortfall"], 2)
        with self.assertRaises(ValueError):
            FC.first_k_selection([(0, True), (2, True)], 3)  # out-of-order attempt


if __name__ == "__main__":
    unittest.main()
