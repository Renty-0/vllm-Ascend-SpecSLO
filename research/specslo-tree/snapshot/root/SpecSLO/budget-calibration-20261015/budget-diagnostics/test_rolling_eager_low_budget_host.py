"""Stdlib-only scenarios and actual V96 vector arithmetic; never device evidence."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from rolling_eager_low_budget_host import reserve_one_eager3,compile_host_selector,PARENT

BASE=Path('/root/SpecSLO/budget-calibration-20261015')
RAW=BASE/'tree-capacity64-runtime-v96/local-output/capacity64-full463-v1/rank1-full463-actual-trace.json.gz'
RAW_SHA='1a4e670c8af3ab33b2e2e7e6072ba903d68c6258a16140d7e5d2c78dc6120ef9'


class Vector:
    def __init__(self,values):self.values=list(values)
    def __getitem__(self,key):return Vector(self.values[key]) if isinstance(key,slice) else self.values[key]
    def detach(self):return self
    def cpu(self):return self
    def tolist(self):return list(self.values)


def host_original(path,name,namespace):
    tree=ast.parse(Path(path).read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]


def fixture(normal_budgets=(1,2,3,4,5),parent_budget=1,repairs=None):
    prefix=100;n=len(normal_budgets);owner=n;repairs=[0]*n if repairs is None else repairs
    def plan(b,p=prefix):return NS(width=2,depth=4,candidate_budget=b,prefix_len=p,
        parent_indices=Vector([-1,0,1,2,-1][:b]),positions=Vector([p]+[p+min(i+1,4) for i in range(b)]))
    parent=plan(parent_budget);pathlen=min(parent_budget,4);parent.positions=Vector([prefix]+[prefix+i+1 for i in range(pathlen)]+([prefix+1] if parent_budget==5 else []))
    plans=[plan(b) for b in normal_budgets]+[plan(3,prefix+pathlen+1)]
    roots=[10+i for i in range(n)]+[99];rows=[101+i for i in range(parent_budget)]
    known={i:dict(safe_before=prefix-1,tokens={prefix-1:7 if not r else 8}) for i,r in enumerate(repairs)}
    known[owner]=dict(safe_before=prefix,tokens={prefix:99,**{prefix+j:rows[j-1] for j in range(1,pathlen)}})
    engine=NS(config=NS(enable_prefix_caching=False,max_num_seqs=64,max_model_len=4096),
        model=NS(context=NS(size=1)),graph_runner=NS(max_graph_tokens=128),_dynamic_tree_known_KV=known)
    return engine,plans,roots,list(range(n+1)),{owner:(parent,rows)},[[7,roots[i]] for i in range(n)]+[()]


class LowBudgetHost(unittest.TestCase):
    def setUp(self):
        fast=host_original(BASE/'tree-s35-production-v101/fast_tree_draft_v5.py','missing_committed_suffix',{})
        primary=host_original('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/tree.py','tree_primary_path',{'TreeSpeculationPlan':object})
        self.enterContext(patch.dict(sys.modules,{'fast_tree_draft_v5':NS(missing_committed_suffix=fast),
            'vllm_ascend.spec_decode.pearl.tree':NS(tree_primary_path=primary)}))

    def test_original_complete_selector_AST_except_three_declared_guards(self):
        select,proof=compile_host_selector()
        self.assertTrue(proof['all_other_selector_AST_unchanged']);self.assertFalse(proof['installed'])
        self.assertFalse(proof['actual_NPU_qualification'])

    def test_actual_function_parent_B1_to5_and_ragged_zero_one_repairs(self):
        select,_=compile_host_selector()
        for parent_b in range(1,6):
            case=fixture(parent_budget=parent_b,repairs=[0,1,0,1,0]);out=select(*case)
            self.assertIsNotNone(out);self.assertEqual(out.normal_rows,tuple(range(5)))
            self.assertEqual(out.eager_rows,(5,));self.assertEqual(len(out.entries),6)
            self.assertEqual([len(e[3]) for e in out.entries[:5]],[0,1,0,1,0])
            self.assertEqual(out.tail_prefixes[-1],100+min(parent_b,4)+1)

    def test_all_B_le3_and_unknown_normal_ledger_fallback_without_mutation(self):
        select,_=compile_host_selector()
        for change in ('all_short','unknown'):
            case=fixture(normal_budgets=(1,2,3) if change=='all_short' else (1,4))
            if change=='unknown':case[0]._dynamic_tree_known_KV.pop(0)
            before=json.dumps(case[0]._dynamic_tree_known_KV,sort_keys=True)
            self.assertIsNone(select(*case));self.assertEqual(json.dumps(case[0]._dynamic_tree_known_KV,sort_keys=True),before)

    def test_parent_dependency_frontier_position_and_stale_root_reject(self):
        select,_=compile_host_selector()
        for change in ('prefix','root','spine','boundary'):
            case=fixture(parent_budget=3);engine,plans,roots,ids,sources,catchups=case
            if change=='prefix':plans[-1].prefix_len+=1
            elif change=='root':engine._dynamic_tree_known_KV[ids[-1]]['tokens'][100]=0
            elif change=='spine':engine._dynamic_tree_known_KV[ids[-1]]['tokens'][101]=0
            else:engine._dynamic_tree_known_KV[ids[-1]]['safe_before']=101
            with self.subTest(change=change),self.assertRaises(RuntimeError):select(*case)

    def test_C96_reserve_is_explicit_normal_quality_debit_never_extra_budget(self):
        normal={i:3 for i in range(32)};normal[0]=4;normal[1]=2
        out=reserve_one_eager3(normal,eager_owner=32,reduction_order=range(32))
        self.assertEqual(out['normal_candidates_debited'],3);self.assertEqual(out['combined_C'],96)
        self.assertEqual(out['normal_budgets'][0],4);self.assertEqual(sum(out['normal_budgets'].values()),93)
        self.assertFalse(out['NPU_qualified']);self.assertFalse(out['supplied_eligibility_and_reduction_order_are_actual_measurements'])
        spare=reserve_one_eager3({0:5,1:5},eager_owner=2,reduction_order=[0,1])
        self.assertEqual(spare['normal_candidates_debited'],0)
        for normal,order in (({0:3},[0]),({0:True,1:4},[0,1]),({0:5},[])):
            with self.assertRaises(ValueError):reserve_one_eager3(normal,eager_owner=32,reduction_order=order)

    def test_all1042_actual_N32_vectors_can_fund_E1B3_only_with_three_candidate_debit(self):
        raw=RAW.read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),RAW_SHA)
        trace=json.loads(gzip.decompress(raw));records=trace['normal_target_calls']+[
            dict(N=len(e['target_ids']),budgets=e['budgets']) for e in trace['logical_events'] if e['kind']=='plan']
        count=0
        for record in records:
            if record['N']!=32:continue
            normal=dict(enumerate(record['budgets']));self.assertEqual(sum(normal.values()),96)
            out=reserve_one_eager3(normal,eager_owner=32,reduction_order=range(32))
            self.assertEqual(out['normal_candidates_debited'],3);self.assertEqual(out['combined_C'],96)
            self.assertEqual(out['combined_draft_rows'],33);self.assertLessEqual(out['normal_target_queries'],128)
            self.assertGreaterEqual(max(out['normal_budgets'].values()),4);count+=1
        self.assertEqual(count,1042)


if __name__=='__main__':unittest.main(verbosity=2)
