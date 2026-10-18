"""Untimed install/rollback of typed allocation, fixed mapper and CPU contexts."""
from pathlib import Path
import json
from dataclasses import replace
from config_snapshot_v138 import snapshot_config
from cpu_default_device_cleanup_host import (observe_cpu_device_context,restore_redundant_cpu_device_context,reinstall_redundant_cpu_device_context)
from split_shape_v132 import shape,filter_eager_parents,refine_draft_groups
from complete_search import envelope as complete
P=Path(__file__).resolve().parent
BASE_MAPPER_PARTITIONS=frozenset((16,q,'causal') for q in range(59,65))
OPTIONAL_LARGE_PARTITIONS=frozenset((n,n*4,'causal') for n in range(27,31))
def mapper_partitions(configuration):
    enabled=configuration['mapper_extra_large_partitions_enabled']
    if type(enabled) is not bool:raise RuntimeError('V132 optional mapper selector must be an explicit boolean')
    return BASE_MAPPER_PARTITIONS | (OPTIONAL_LARGE_PARTITIONS if enabled else frozenset())
MAPPER_PARTITIONS=mapper_partitions(json.loads((P/'CANDIDATE_V132.json').read_text()))

class CandidateControls:
    def __init__(self,*,native,rhythm,policy,torch,constructor_state,is_draft,configuration):
        self.native,self.rhythm,self.policy,self.torch,self.state=native,rhythm,policy,torch,constructor_state
        self.is_draft=bool(is_draft);self.config=dict(configuration);self.mode='original';self.installed=False;self.engine=None;self.original_config=None;self.candidate_config=None
        if set(self.config)!={'target_cpu_context_clean','draft_cpu_context_clean'} or any(type(v) is not bool for v in self.config.values()):
            raise RuntimeError('V132 requires two explicit role CPU context booleans')
        self.original_shape=rhythm.SpecRhythmBudgetShaper.shape
        cells=dict(zip(self.original_shape.__code__.co_freevars,self.original_shape.__closure__ or ()))
        if 'policy' not in cells or cells['policy'].cell_contents is not policy:raise RuntimeError('V132 shape/metrics must keep the same original policy object')
        def candidate_shape(instance,**kwargs):return shape(policy,instance,**kwargs)
        self.candidate_shape=candidate_shape
        self.original_allocate=type(policy).allocate
        import live_decode_patch,target_layout
        self.live,self.layout=live_decode_patch,target_layout
        self.original_builder=live_decode_patch.build_live_runtime
        self.original_envelope=target_layout.envelope
        self.previous_globals={name:native.__dict__.get(name) for name in ('_v132_filter_eager_parents','_v123_refine_draft_groups')}
        if any(value is not None for value in self.previous_globals.values()):raise RuntimeError('V132 Native delegate names already occupied')
        def mapper(n,q,f):return complete(n,q,f) if (n,q,f) in MAPPER_PARTITIONS else self.original_envelope(n,q,f)
        self.mapper=mapper
        self.observe()

    def observe(self):
        clean=self.mode=='candidate' and self.config['draft_cpu_context_clean' if self.is_draft else 'target_cpu_context_clean']
        cpu=observe_cpu_device_context(self.torch,self.state,redundant_cpu=not clean)
        expected=self.candidate_shape if self.mode=='candidate' else self.original_shape
        if not cpu['passed'] or self.rhythm.SpecRhythmBudgetShaper.shape is not expected or type(self.policy).allocate is not self.original_allocate:
            raise RuntimeError('V132 original policy/context/shape ownership differs')
        if self.mode=='candidate' and (self.layout.envelope is not self.mapper or self.native._v132_filter_eager_parents is not filter_eager_parents
            or self.native._v123_refine_draft_groups is not refine_draft_groups):raise RuntimeError('V132 runtime or mapper delegate ownership differs')
        config=None if self.engine is None else self.engine.config
        expected_config=self.candidate_config if self.mode=='candidate' else self.original_config
        if self.engine is not None and config is not expected_config:raise RuntimeError('V132 actual config ownership changed')
        effective=None if config is None else dict(draft_token_budget=config.spec_rhythm_draft_token_budget,eager_token_cap=config.spec_rhythm_max_eager_tokens,eager_reserve_tokens=config.spec_rhythm_eager_reserve_tokens)
        return dict(actual_effective_config=effective,mode=self.mode,is_draft=self.is_draft,CPU=cpu,CPU_clean=clean,typed_draft192=self.mode=='candidate',
            normal_limit=96,future_eager_limit=96,global_draft_limit=192,current_target_C=96,current_target_Q=128,current_target_N=32,
            normal_N_threshold=8,normal_N_gt8_cap=4,normal_N_le8_cap=5,eager_exact_publish_B=3,eligible_parent_B=[3],
            partial_eager_grants_deferred=True,original_policy_object_preserved=True,original_allocate_unchanged=True,
            mapper_partitions=sorted(MAPPER_PARTITIONS),Native_flags_assigned=False,global_factory_equivalence_proved=False)

    def select(self,mode,*,engine):
        if mode not in ('original','candidate'):raise RuntimeError('V132 unknown control mode')
        if engine.cache_allocation is not None or engine.cache_block_tables is not None:raise RuntimeError('V132 mode switch requires original full cache release')
        self.observe()
        if self.engine is None:
            self.engine=engine;self.original_config=engine.config
            if (engine.config.spec_rhythm_max_eager_tokens!=5 or engine.config.spec_rhythm_draft_token_budget is not None or engine.config.spec_rhythm_eager_reserve_tokens!=0):raise RuntimeError('V132 constructor must retain the original fixed inventory configuration')
            self.candidate_config=replace(engine.config,spec_rhythm_draft_token_budget=192,spec_rhythm_max_eager_tokens=3,spec_rhythm_eager_reserve_tokens=32)
            before,after=snapshot_config(self.original_config),snapshot_config(self.candidate_config)
            changed={name:(before[name],after[name]) for name in before if before[name]!=after[name]}
            if changed!={'spec_rhythm_draft_token_budget':(None,192),'spec_rhythm_max_eager_tokens':(5,3),'spec_rhythm_eager_reserve_tokens':(0,32)}:raise RuntimeError('V132 config delta exceeded three explicit draft fields')
            self.config_snapshots=dict(original=before,candidate=after,delta=changed)
        elif self.engine is not engine:raise RuntimeError('V132 control engine changed')
        if mode=='candidate' and not self.installed:
            import runtime_adapter_v132
            self.rhythm.SpecRhythmBudgetShaper.shape=self.candidate_shape
            self.native._v132_filter_eager_parents=filter_eager_parents
            self.native._v123_refine_draft_groups=refine_draft_groups
            # build_live_runtime retains its original ABI and appends the
            # reviewed adapter only after all original AST edits succeed.
            self.layout.envelope=self.mapper;self.installed=True;engine.config=self.candidate_config
        elif mode=='original' and self.installed:
            self.rhythm.SpecRhythmBudgetShaper.shape=self.original_shape
            for name in self.previous_globals:delattr(self.native,name)
            self.layout.envelope=self.original_envelope;self.installed=False;engine.config=self.original_config
        want_clean=mode=='candidate' and self.config['draft_cpu_context_clean' if self.is_draft else 'target_cpu_context_clean']
        if not observe_cpu_device_context(self.torch,self.state,redundant_cpu=not want_clean)['passed']:
            result=(restore_redundant_cpu_device_context if want_clean else reinstall_redundant_cpu_device_context)(self.torch,self.state)
            if result['status']!=('cleaned' if want_clean else 'reinstalled'):raise RuntimeError('V132 strict CPU context switch refused')
        self.mode=mode
        return self.observe()
