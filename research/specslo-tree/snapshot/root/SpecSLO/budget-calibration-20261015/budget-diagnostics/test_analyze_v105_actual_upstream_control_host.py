"""HOST-only parser negatives; no actual audit or Torch in these tests."""
import io
from pathlib import Path
import tempfile
import tarfile
import unittest
from unittest.mock import patch
import analyze_v105_actual_upstream_control as a


class ExternalHOST(unittest.TestCase):
    def test_nonterminal_and_failed_refuse_before_raw_or_Torch(self):
        for status in ('running','failed',None):
            with patch.object(a,'read',return_value=dict(status=status)),patch.object(a,'source',side_effect=AssertionError('too early')):
                with self.assertRaisesRegex(RuntimeError,'complete actual control'):a.run('HOST')

    def test_source_identity_changes_refuse(self):
        with patch.object(a,'sha',return_value='changed'):
            with self.assertRaisesRegex(RuntimeError,'reviewed CPU/freeze'):a.source({},'HOST')

    def test_archive_duplicate_or_changed_bytes_refused(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);f=p/'source.tar.gz'
            for names in (['x.py'],['x.py','x.py']):
                with tarfile.open(f,'w:gz') as t:
                    for name in names:
                        m=tarfile.TarInfo(name);m.size=4;t.addfile(m,io.BytesIO(b'good'))
                with patch.object(a,'P',p):
                    expected={str(p/'x.py'):hashlib.sha256(b'good').hexdigest()}
                    if len(names)==1:self.assertEqual(a.archive(f,expected)['members'],1)
                    else:
                        with self.assertRaises(RuntimeError):a.archive(f,expected)
                    with self.assertRaises(RuntimeError):a.archive(f,{str(p/'x.py'):'bad'})

    def test_frozen_parser_keeps_missing_Native21_fail_closed(self):
        parser=a.module('audit_upstream_v105');contract=a.module('upstream_contract_v105')
        good={k:0 for k in contract.NATIVE_COUNTERS};row=dict(before=dict(good),after=dict(good),delta=dict(good))
        parser.native_delta(row,zero=True)
        row['before'].pop(contract.NATIVE_COUNTERS[0])
        with self.assertRaises(RuntimeError):parser.native_delta(row,zero=True)


if __name__=='__main__':unittest.main(verbosity=2)
