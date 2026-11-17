"""Prepared geometry coefficient reuse for analytic, not Panda, K4."""
import unittest
from dataclasses import replace
from fractions import Fraction as F
from curobo_certify.k4_reference import PlanarTwoLink, PathSpan, I, replay_torque_leaves
from curobo_certify.k4_affine import (prepare_geometry, verify_prepared_geometry,
    certify_with_prepared_geometry, geometry_digest)

class GeometryReuseTest(unittest.TestCase):
    def setUp(self):
        self.model = PlanarTwoLink(1,1,1,F(1,2),F(1,2),0,0,0)
        self.s1 = PathSpan.from_values(0,1,[[0,0,F(1,2)],[0]],[1])
        self.s2 = PathSpan.from_values(0,1,[[0,0,F(1,2)],[0]],[F(1,2)])
        self.limits = [(-3,3),(-1,1)]

    def test_reuse_fixed_path_vary_timing(self):
        box=prepare_geometry([self.s1],self.model,depth=3)
        self.assertTrue(verify_prepared_geometry([self.s1],self.model,box))
        self.assertEqual(geometry_digest([self.s1]),geometry_digest([self.s2]))
        for s in (self.s1,self.s2):
            r=certify_with_prepared_geometry([s],self.model,self.limits,box)
            self.assertEqual(r.status,"CERTIFIED")
            self.assertTrue(replay_torque_leaves([s],self.model,self.limits,r))

    def test_invalid_model_geometry_or_boxes_rejected(self):
        box=prepare_geometry([self.s1],self.model,depth=2)
        changed=PathSpan.from_values(0,1,[[0,0,1],[0]],[1])
        self.assertFalse(verify_prepared_geometry([changed],self.model,box))
        self.assertEqual(certify_with_prepared_geometry([changed],self.model,self.limits,box).status,
                         "INVALID_PREPARED_GEOMETRY")
        bad_leaf=replace(box.leaves[0],A=(I.point(0),I.point(0)))
        broken=replace(box,leaves=(bad_leaf,)+box.leaves[1:])
        self.assertFalse(verify_prepared_geometry([self.s1],self.model,broken))

    def test_tighter_limit_fails_closed(self):
        box=prepare_geometry([self.s1],self.model,depth=2)
        result=certify_with_prepared_geometry([self.s1],self.model,[(-1,1),(-1,1)],box)
        self.assertEqual(result.status,"UNKNOWN_PREPARED_ENCLOSURE")

    def test_budget_and_partition(self):
        with self.assertRaises(ValueError):
            prepare_geometry([self.s1],self.model,depth=15)
        with self.assertRaises(ValueError):
            prepare_geometry([self.s1],self.model,depth=14,max_leaves=8)
        box=prepare_geometry([self.s1],self.model,depth=3)
        self.assertFalse(verify_prepared_geometry([self.s1],self.model,replace(box,leaves=box.leaves[:-1])))

if __name__=="__main__":unittest.main()
