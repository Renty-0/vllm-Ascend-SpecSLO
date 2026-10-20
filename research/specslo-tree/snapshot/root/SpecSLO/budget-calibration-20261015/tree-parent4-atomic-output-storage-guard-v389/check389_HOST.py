"""Reproduce the actual stat-vs-replace race and retain strict outside errors."""
import errno
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import guard389


def race(root,live,filename,allowed):
    path=root/filename;path.write_bytes(b'1234567')
    original=Path.stat;fired=[]
    def stat(item,*args,**kwargs):
        if item==path and not fired and kwargs.get('follow_symlinks',True):
            fired.append(True);path.unlink()
        return original(item,*args,**kwargs)
    try:
        with patch.object(Path,'stat',stat):
            result=guard389.safe_storage177.bytes_under(root,live_roots=(live,),require=guard389.require)
        assert allowed and result==0 and fired
    except FileNotFoundError as error:
        assert not allowed and error.errno==errno.ENOENT and fired


def main():
    with tempfile.TemporaryDirectory(prefix='HOST389-',dir=Path(__file__).parent)as directory:
        root=Path(directory);live=root/'actual';live.mkdir();outside=root/'immutable';outside.mkdir()
        race(live,live,'prefill-rank2.tmp',True)
        race(live,live,'prefill-rank2.json',False)
        race(outside,live,'immutable.tmp',False)
        (live/'completed.json').write_bytes(b'12345')
        assert guard389.safe_storage177.bytes_under(live,live_roots=(live,),require=guard389.require)==5
        (live/'alias').symlink_to(live/'completed.json')
        try:guard389.safe_storage177.bytes_under(live,live_roots=(live,),require=guard389.require)
        except RuntimeError:pass
        else:raise AssertionError('symlink must reject')
    ns=guard389.execution_namespace()
    assert ns['storage']is guard389.storage and ns['execute_locked'].__globals__ is ns
    assert 'torch'not in __import__('sys').modules
    value=dict(status='PASS_HOST389_REAL_ATOMIC_STAT_RACE_AND_STRICT_NEGATIVES',NPU_used=False,
        actual_tmp_race_reproduced=True,non_tmp_error_preserved=True,outside_live_tmp_error_preserved=True,
        symlink_rejection_preserved=True,real_file_byte_count=True,model_runtime_changed=False,Raw=None,Goodput=None)
    (Path(__file__).parent/'HOST389_RESULT.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))


if __name__=='__main__':main()
