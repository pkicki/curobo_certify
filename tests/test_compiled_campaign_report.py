import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from render_compiled_campaign_report import render

class CompiledCampaignReportTests(unittest.TestCase):
    def test_offline_report_embeds_all_research_views(self):
        data={'scope':'EXPLORATORY_HULL_ONLY_PLUS_MODEL_PROOF_AND_DEVICE_DIAGNOSTIC',
              'paths':[{'id':'P0','source_sha':'a','model_sha':'b','upstream':'pin',
                'model':{'spans':[{'q':[['0']]}]},'operands':[],
                'benchmark':{'scope':'EXPLORATORY_FROZEN_MODEL_CPU_ONLY','case_summary':{},'raw':[]},
                'tightness':{'residual_records':[]},'model_proof':'K3 model-only'}]}
        out=render(data)
        for token in ('BLOCKED_NATIVE_LINKAGE','initial Bernstein hull','captured s',
                      'Export selected evidence JSON','EXPLORATORY / PARTIAL ONLY'):
            self.assertIn(token,out)
        self.assertNotIn('https://',out)
    def test_rejects_wrong_scope(self):
        with self.assertRaises(ValueError):render({'scope':'CERTIFIED'})
if __name__=='__main__':unittest.main()
