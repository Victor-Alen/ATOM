"""Check immutable asset integrity and offline/online resolution contracts."""
from pathlib import Path
import hashlib,io,json,os,sys,tempfile,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import mtp_asset

class AssetTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
  self.head=b'fixture checkpoint bytes';self.digest=hashlib.sha256(self.head).hexdigest()
  self.manifest={'head_sha256':self.digest}
  p=self.root/'deployment/mtp-model';p.mkdir(parents=True)
  (p/'official-checkpoint-manifest.json').write_text(json.dumps(dict(repo='owner/model',revision='fixed-commit',head=dict(file='model-mtp.safetensors',bytes=len(self.head),sha256=self.digest))))
  self.env=patch.dict(os.environ,{},clear=True);self.env.start()
 def tearDown(self):self.env.stop();self.tmp.cleanup()
 def bundled(self,data):
  p=self.root/'deployment/model-assets/model-mtp.safetensors';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);return p
 def test_offline_bundled(self):
  p=self.bundled(self.head)
  with patch.object(mtp_asset.urllib.request,'urlopen',side_effect=AssertionError('offline path used network')):
   self.assertEqual(mtp_asset.resolve_head(self.root,self.manifest),p)
 def test_explicit_corruption_fails_without_fallback(self):
  self.bundled(self.head);p=self.root/'wrong';p.write_bytes(b'bad');os.environ['MTP_HEAD_PATH']=str(p)
  with self.assertRaisesRegex(RuntimeError,'SHA256'):mtp_asset.resolve_head(self.root,self.manifest)
 def test_offline_missing_no_network(self):
  os.environ['MTP_DOWNLOAD']='0'
  with patch.object(mtp_asset.urllib.request,'urlopen',side_effect=AssertionError('offline used network')):
   with self.assertRaisesRegex(RuntimeError,'MTP_DOWNLOAD=0'):mtp_asset.resolve_head(self.root,self.manifest)
 def test_download_hash_and_cache(self):
  with patch.object(mtp_asset.urllib.request,'urlopen',return_value=io.BytesIO(self.head)) as request:
   p=mtp_asset.resolve_head(self.root,self.manifest)
   self.assertIn('/resolve/fixed-commit/',request.call_args.args[0].full_url)
  self.assertEqual(p.read_bytes(),self.head)
  with patch.object(mtp_asset.urllib.request,'urlopen',side_effect=AssertionError('cache used network')):
   self.assertEqual(mtp_asset.resolve_head(self.root,self.manifest),p)
  receipt=json.loads(p.with_name('download-receipt.json').read_text());self.assertEqual(receipt['sha256'],self.digest)
 def test_incomplete_corrupt_oversized_never_published(self):
  for data in (self.head[:-1],b'x'*len(self.head),self.head+b'extra'):
   with self.subTest(data=data),patch.object(mtp_asset.urllib.request,'urlopen',side_effect=lambda *a,**k:io.BytesIO(data)):
    with self.assertRaisesRegex(RuntimeError,'download failed'):mtp_asset.resolve_head(self.root,self.manifest)
    cache=self.root/'.runtime/model-assets'
    self.assertFalse((cache/'model-mtp.safetensors').exists())
    self.assertEqual(list(cache.glob('.mtp-download-*')),[])

if __name__=='__main__':unittest.main(verbosity=2)
