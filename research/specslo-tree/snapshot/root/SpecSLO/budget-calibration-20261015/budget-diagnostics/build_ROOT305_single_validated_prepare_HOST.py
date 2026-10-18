"""New HOST backend prototype; frozen296 and active299 are never changed."""
from pathlib import Path

B = Path('/root/SpecSLO/budget-calibration-20261015')
P = B / 'tree-single-validated-prefix-prepare-host-v305'


def main():
    P.mkdir(exist_ok=False)
    source = (B / 'tree-concrete-device-span-dual-cache-target-cost-v296/static_state296.py').read_text()
    source = source.replace('296', '305')

    def replace(before, after, count=1):
        nonlocal source
        assert source.count(before) == count, (before, source.count(before))
        source = source.replace(before, after)

    replace('state.copy305_layout_HOST = None',
            'state.copy305_layout_HOST = None\n    state.copy305_contract = None')
    replace("from prefix278 import select_grid\n            state.copy280_bucket=select_grid(desc['source'],desc['dest'],rows=544*128)\n            # select_grid already executes the identical complete contract278.checked.",
            "from descriptors304 import CheckedRows\n            validated=CheckedRows(desc['source'],desc['dest'],rows=544*128)\n            desc['source']=validated.source\n            desc['dest']=validated.destination\n            state.copy305_contract=validated\n            state.copy280_bucket=validated.grid\n            # Immutable source maps retain the original full anti-alias proof.")
    replace('state.copy280_bucket=0\n        return desc',
            'state.copy280_bucket=0\n            state.copy305_contract=None\n        return desc')
    replace('from split288 import split', 'from descriptors304 import separate')
    replace('    state.prefix305_source=torch.empty(1792,dtype=torch.long,device=engine.device)\n', '')
    replace("proposal=split(desc['source'],desc['dest'],query_slots)\n            state.prefix305_source.copy_(torch.tensor(proposal['prefix_source'],dtype=torch.long,device='cpu'))",
            "if desc['source']is not state.copy305_contract.source or desc['dest']is not state.copy305_contract.destination:\n                raise RuntimeError('ROOT305 original H2D maps must be the exact immutable checked maps')\n            proposal=separate(state.copy305_contract,query_slots)")
    replace("            from spans292 import pack as pack_spans\n            spans=pack_spans(proposal['prefix_source'],proposal['prefix_destination'])\n            state.prefix305_spans.copy_(torch.tensor(spans['spans'],dtype=torch.long,device='cpu'))",
            "            state.prefix305_spans.copy_(torch.tensor(proposal['spans'],dtype=torch.long,device='cpu'))")
    replace('state.prefix305_source.data_ptr(),', '', count=2)
    replace('from device305 import resolve_cache_device', 'from device296 import resolve_cache_device')
    replace('from capture_key305 import install as install_key', 'from capture_key296 import install as install_key')
    compile(source, str(P / 'static_state305.py'), 'exec')
    (P / 'static_state305.py').write_text(source)
    print('New305 HOST prepare source written; no runtime installed or model launched')


if __name__ == '__main__':
    main()
