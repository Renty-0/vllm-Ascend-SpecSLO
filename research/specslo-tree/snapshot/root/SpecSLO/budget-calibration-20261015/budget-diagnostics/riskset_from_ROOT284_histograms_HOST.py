"""Recover exact risk-set counts from retained aggregate verdict histograms.

No token strings or unobserved continuation outcomes are reconstructed. The
input is restricted to the root-alternate primary4 topology used by284.
"""
import copy,json

def blank():
    return dict(primary=[dict(samples=0,hits=0)for _ in range(4)],
        alternate_first=dict(samples=0,hits=0),
        alternate_continuation=[dict(samples=0,hits=0)for _ in range(3)],
        by_budget={})

def recover(by_budget):
    out=blank()
    for key,row in by_budget.items():
        b=int(key);n=row['samples'];h=row['delivered_histogram'];side=row['side_histogram']
        assert 1<=b<=8 and len(h)==6 and len(side)==5 and n>0 and sum(h)==sum(side)==n and h[0]==0
        assert all(type(v)is int and v>=0 for v in h+side)
        assert sum(i*v for i,v in enumerate(h))==row['delivered_sum']
        assert sum(i*v for i,v in enumerate(side))==row['accepted_side_nodes']
        assert all(side[i]==0 for i in range(max(0,b-4)+1,5))
        path=[0]*5
        for length in range(1,min(4,b)+1):
            path[length]=h[length+1]-side[length]
            assert path[length]>=0
        path[0]=n-sum(path[1:]);assert path[0]==h[1]+sum(side[1:])
        for depth in range(1,min(4,b)+1):
            layer=out['primary'][depth-1];layer['samples']+=sum(path[depth-1:]);layer['hits']+=sum(path[depth:])
        if b>=5:
            out['alternate_first']['samples']+=path[0];out['alternate_first']['hits']+=sum(side[1:])
            for depth,layer in enumerate(out['alternate_continuation'],2):
                if b-4>=depth:
                    layer['samples']+=sum(side[depth-1:]);layer['hits']+=sum(side[depth:])
        out['by_budget'][str(b)]=dict(samples=n,mean_delivered=row['delivered_sum']/n,
            mean_side=row['accepted_side_nodes']/n,
            mean_same_prefix_B4=row['shadow_B4_sum']/n if b>=4 else None,
            primary_path_counts=path,alternate_path_counts=list(side))
    return out

def merge(rows):
    out=blank()
    for row in rows:
        for name in('primary','alternate_continuation'):
            for dest,src in zip(out[name],row[name],strict=True):
                for k in('samples','hits'):dest[k]+=src[k]
        for k in('samples','hits'):out['alternate_first'][k]+=row['alternate_first'][k]
    return out

def main():
    # Every possible accepted primary/alternate prefix in the actual padded
    # verdict is aggregated independently, then recovered from only histograms.
    cases=0
    for b in range(1,9):
        paths=[list(range(k))for k in range(min(4,b)+1)]
        if b>=5:paths += [list(range(4,4+k))for k in range(1,b-3)]
        hist=dict(samples=0,delivered_sum=0,accepted_side_nodes=0,shadow_B4_sum=0,
            delivered_histogram=[0]*6,side_histogram=[0]*5)
        expected=blank()
        for path in paths:
            copies=1+len(path);actual=len(path)+1;side=sum(v>=4 for v in path);primary=len(path)if not side else 0
            for _ in range(copies):
                hist['samples']+=1;hist['delivered_sum']+=actual;hist['accepted_side_nodes']+=side
                if b>=4:hist['shadow_B4_sum']+=actual-side
                hist['delivered_histogram'][actual]+=1;hist['side_histogram'][side]+=1
                for depth,layer in enumerate(expected['primary'],1):
                    if depth<=min(b,4)and primary>=depth-1:
                        layer['samples']+=1;layer['hits']+=int(primary>=depth)
                if b>=5 and primary==0:
                    expected['alternate_first']['samples']+=1;expected['alternate_first']['hits']+=int(side>0)
                for depth,layer in enumerate(expected['alternate_continuation'],2):
                    if b-4>=depth and side>=depth-1:
                        layer['samples']+=1;layer['hits']+=int(side>=depth)
                cases+=1
        actual=recover({str(b):hist})
        for name in('primary','alternate_first','alternate_continuation'):assert actual[name]==expected[name],(b,name)
    print(json.dumps(dict(status='HOST_exact_root_risksets_recovered_from_histograms',synthetic_observations=cases,all_B1to8=True,no_unobserved_outcomes_filled=True,NPU_used=False,empirical_machine_g=False)))
if __name__=='__main__':main()
