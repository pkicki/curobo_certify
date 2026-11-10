from __future__ import annotations

import unittest

from curobo_certify.query_pipeline import pilot_queries, verify_query


class NativeQueryManifestTests(unittest.TestCase):
    def test_seeded_queries_are_reproducible_and_hashed(self):
        first = pilot_queries()
        self.assertEqual(first, pilot_queries())
        self.assertEqual(len(first), 16)
        self.assertEqual(sum(row["robot"] == "panda" for row in first), 8)
        self.assertEqual(sum(row["robot"] == "ur5e" for row in first), 8)
        self.assertTrue(all(verify_query(row) for row in first))
        changed = dict(first[0], scene_id="collision_test")
        self.assertFalse(verify_query(changed))

    def test_independent_ids_and_strata(self):
        rows = pilot_queries()
        self.assertEqual(len({row["source_id"] for row in rows}), 16)
        self.assertEqual({row["scene_stratum"] for row in rows},
                         {"obstacle_free", "moderately_cluttered", "difficult"})
        self.assertTrue(all(len(row["goal_delta_rad"]) == (7 if row["robot"] == "panda" else 6)
                            for row in rows))
        panda = [row for row in rows if row["robot"] == "panda"]
        self.assertTrue(all(len(row["start_joint_positions_rad"]) == 7 and
                            len(row["goal_joint_positions_rad"]) == 7 and
                            row["query_resolution"] == "FROZEN_EXPLICIT_CSPACE" for row in panda))
        ur5 = [row for row in rows if row["robot"] == "ur5e"]
        self.assertTrue(all(row["start_joint_positions_rad"] is None and
                            row["query_resolution"] == "BLOCKED_NO_VERIFIED_MODEL_ASSET" for row in ur5))


if __name__ == "__main__":
    unittest.main()
