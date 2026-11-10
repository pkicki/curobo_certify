import hashlib, importlib.util, json, tempfile, unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('freeze_corpus','scripts/freeze_corpus.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
class FreezeTests(unittest.TestCase):
 def row(self,root,path_id='p1',split='Dev'):
  arr=root/'q.bin'; arr.write_bytes(b'immutable'); return {'path_id':path_id,'source_id':'query-1','scene_id':'scene-1','robot':'panda','split':split,'arrays':{'q':{'path':'q.bin','sha256':hashlib.sha256(b'immutable').hexdigest()}},'provenance':'public fixture'}
 def test_freezes_hashes_and_counts(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); src=root/'source.jsonl'; src.write_text(json.dumps(self.row(root))+'\n'); out=root/'frozen'; m=mod.freeze(src,out)
   self.assertTrue(m['frozen']); self.assertEqual(m['split_counts']['Dev'],1); self.assertEqual(json.loads((out/'manifest.json').read_text())['records'][0]['arrays']['q']['sha256'],hashlib.sha256(b'immutable').hexdigest())
 def test_rejects_source_scene_split_leakage(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); a=self.row(root,'p1','Dev'); b=self.row(root,'p2','Holdout'); (root/'source.jsonl').write_text(json.dumps(a)+'\n'+json.dumps(b)+'\n')
   with self.assertRaisesRegex(ValueError,'leakage'): mod.freeze(root/'source.jsonl',root/'out')
 def test_rejects_array_mutation(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); row=self.row(root); (root/'q.bin').write_bytes(b'changed'); (root/'source.jsonl').write_text(json.dumps(row)+'\n')
   with self.assertRaisesRegex(ValueError,'SHA256 mismatch'): mod.freeze(root/'source.jsonl',root/'out')
if __name__=='__main__': unittest.main()
