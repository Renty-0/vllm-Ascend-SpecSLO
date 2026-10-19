"""Read-only comparison of complete official0.21 and archived0.23 repeats."""
import datetime
import hashlib
import json
import statistics
from pathlib import Path

ROOT=Path('/root/SpecSLO/official-v021-baseline-20261019')
OLD=Path('/root/SpecSLO/tpot-report-20261015/official-confirm3/summary.json')
KINDS=('tight','normal','loose')

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def spread(values):return (max(values)-min(values))/statistics.median(values)

def aggregate(rows):
    goodput=[r['Goodput']for r in rows]
    raw=[r['Raw']for r in rows]
    return dict(Goodput_all=goodput,Goodput_median=statistics.median(goodput),
        Goodput_min=min(goodput),Goodput_max=max(goodput),Goodput_range_over_median=spread(goodput),
        Goodput_sample_CV=statistics.stdev(goodput)/statistics.mean(goodput),
        Raw_all=raw,Raw_median=statistics.median(raw),Raw_range_over_median=spread(raw),
        TPOT_median_range_over_median={k:spread([r['groups'][k]['median']for r in rows])for k in KINDS},
        representative_round=rows[goodput.index(statistics.median(goodput))],rounds=rows)

def main():
    receipt=ROOT/'controls-v2/receipt.json'
    guard=read(receipt)
    assert guard['status']=='complete' and not guard['errors'] and len(guard['rounds'])==3
    rows=[]
    for row in guard['rounds']:
        assert row['status']=='complete' and not row['errors']
        assert row['sources_unchanged'] and not row['owned_remaining'] and row['devices_after']['all_idle']
        run=ROOT/'runs'/row['tag']
        audit=read(run/'independent-audit.json')
        assert audit['passed'] and not audit['errors'] and audit['MC2_enabled']
        assert audit['selected_MC2_modules_per_rank']==[128]*4
        assert audit['result_sha256']==sha(run/'result.json')
        groups={k:dict(requests=v['requests'],attained=v['attained'],median=v['TPOT_paper_median_ms'],P90=v['TPOT_paper_P90_ms'])
                for k,v in audit['common_publication_metrics']['by_class'].items()}
        rows.append(dict(tag=row['tag'],Goodput=audit['paper_Goodput'],Raw=audit['raw_throughput'],
            attained=audit['attained_requests'],groups=groups,result=str(run/'result.json'),
            result_sha256=audit['result_sha256'],audit_sha256=sha(run/'independent-audit.json')))
    old=read(OLD)
    assert old['complete_rounds']==old['requested_rounds']==3 and old['all_MC2_ON']
    old_rows=[]
    for row in old['rounds']:
        assert row['audit_passed'] and row['MC2_modules_per_rank']==[128]*4
        assert sha(row['result_path'])==row['result_sha256']
        groups={k:dict(requests=row['groups'][k]['requests'],attained=row['groups'][k]['paper_attained'],
                      median=row['groups'][k]['paper_tpot_ms']['median'],P90=row['groups'][k]['paper_tpot_ms']['P90'])for k in KINDS}
        old_rows.append(dict(tag=row['tag'],Goodput=row['paper_Goodput'],Raw=row['raw_tokens_per_second'],
            attained=row['groups']['all']['paper_attained'],groups=groups,result=row['result_path'],result_sha256=row['result_sha256']))
    new=aggregate(rows);previous=aggregate(old_rows)
    assert new['Goodput_range_over_median']==guard['stability']['Goodput_range_over_median']
    stable=guard['stability']['passed']
    result=dict(schema='official021-vs023-observed-stability.v1',
        Asia_Shanghai=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        official021=new,official023=previous,official021_passes_predeclared_stability=stable,
        official021_observed_Goodput_variation_smaller=new['Goodput_range_over_median']<previous['Goodput_range_over_median'],
        old021_source=dict(plugin_tag='v0.21.0rc1',plugin_commit='80610e4438dba05011b05f89fc45d91e96992671',
            core_tag='v0.21.0',core_commit='ad7125a431e176d4161099480a66f0169609a690',Python='3.11',CANN='9.0',
            physical_cards=[2,5,6,7],source_manifest_sha256=guard['source_manifest_sha256']),
        old023_source=dict(plugin_version='0.23.0.post1',plugin_commit='1cdb8c4',core_version='0.23.0',core_commit='0fc695f',
            Python='3.12',CANN='9.1',physical_cards=old['physical_devices']),
        predeclared_criteria=guard['stability']['criteria'],receipt_sha256=sha(receipt),archived023_summary_sha256=sha(OLD),
        selected_legacy_baseline_if_stable=dict(Goodput=new['Goodput_median'],representative_tag=new['representative_round']['tag'],
            one_point_five_times=new['Goodput_median']*1.5)if stable else None,
        HUST_baseline_run=False,tree_Raw_above_baseline_required=False,
        scope='Three observations per configuration; Python/CANN and visible physical card identifiers differ, so this is not a version-only causal experiment. Final method comparison must rerun on identical current hardware/configuration clocks.')
    path=ROOT/'STABILITY_COMPARISON_V021_V023.json'
    with path.open('x')as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    lines=['# 官方0.21与0.23基线波动比较','',
        '两组均为官方 vLLM Ascend、Qwen3-32B TP4 BF16、MC2 ON；原463请求、同一泊松到达轨迹、每请求256输出token。',
        '0.21指官方插件 v0.21.0rc1（80610e44）与 vLLM v0.21.0（ad7125a4），没有运行 HUST 基线。','',
        '| 官方版本 | 三轮 Goodput token/s | Goodput中位数 | Goodput全距/中位数 | Raw中位数 | Raw全距/中位数 |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for label,data in [('0.21',new),('0.23历史复测',previous)]:
        vals=' / '.join(f'{v:.2f}'for v in data['Goodput_all'])
        lines.append(f"| {label} | {vals} | {data['Goodput_median']:.2f} | {data['Goodput_range_over_median']:.2%} | {data['Raw_median']:.2f} | {data['Raw_range_over_median']:.3%} |")
    lines+=['','全距/中位数 =（三轮最高值－最低值）÷三轮中位数。',
        '','0.21预先规定的稳定条件：Raw全距/中位数≤3%，Goodput≤10%，每类TPOT中位数的全距/中位数≤5%。',
        f"本次0.21是否通过上述条件：{'通过'if stable else'未通过'}。",'']
    if stable:
        lines.append(f"按用户要求，后续选用0.21的Goodput中位数 {new['Goodput_median']:.4f} token/s；1.5倍为 {new['Goodput_median']*1.5:.4f} token/s。Raw、TPOT和达标数均取同一个实际代表轮次 {new['representative_round']['tag']}。")
    else:lines.append('本次不将0.21认定为稳定基线，也不据此下调最终性能目标。')
    lines+=['','| 0.21轮次 | Raw | Goodput | 总体达标 | tight达标 | normal达标 | loose达标 |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        groups=row['groups']
        lines.append(f"| {row['tag']} | {row['Raw']:.2f} | {row['Goodput']:.2f} | {row['attained']}/463 | {groups['tight']['attained']}/271 | {groups['normal']['attained']}/96 | {groups['loose']['attained']}/96 |")
    lines+=['','这里比较的是两组实际重复测量的波动。0.21使用Python3.11/CANN9.0，0.23历史记录使用Python3.12/CANN9.1，可见物理卡编号也不同；不能把差异全部归因于框架版本。最终性能比较须在当前同组卡上重跑选定基线与SpecSLO。',
        '','树形Raw优化继续与已有树形版本比较，不要求树形Raw超过自回归基线；最终仍以完整正确性与Goodput为准。',
        '','数据与逐轮来源见 [STABILITY_COMPARISON_V021_V023.json](STABILITY_COMPARISON_V021_V023.json)。']
    with(ROOT/'STABILITY_COMPARISON_V021_V023_zh.md').open('x')as stream:stream.write('\n'.join(lines)+'\n')
    print(json.dumps({k:result[k]for k in ('official021_passes_predeclared_stability','official021_observed_Goodput_variation_smaller','selected_legacy_baseline_if_stable')},ensure_ascii=False))

if __name__=='__main__':main()
