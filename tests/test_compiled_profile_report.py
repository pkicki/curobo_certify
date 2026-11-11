import json
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from render_compiled_profile_html import render

class ReportTests(unittest.TestCase):
    def test_embeds_only_hull_data(self):
        payload={'experiment':'COMPILED_HULL_STAGE_TIMING_NOT_FULL_CERTIFICATION',
                 'results':[{'model':'test','full_k3_certificate':False,'campaign_eligible':False,'raw':[]}]}
        report=render(payload)
        self.assertIn('EXPLORATORY ONLY',report)
        self.assertNotIn('__DATA__',report)
        self.assertNotIn('https://',report)
    def test_fail_closed_on_bad_scope(self):
        with self.assertRaises(ValueError):render({'experiment':'FULL_CERTIFICATE'})
        with self.assertRaises(ValueError):render({'experiment':'COMPILED_HULL_STAGE_TIMING_NOT_FULL_CERTIFICATION',
                                                  'results':[{'full_k3_certificate':True}]})
if __name__=='__main__':unittest.main()
