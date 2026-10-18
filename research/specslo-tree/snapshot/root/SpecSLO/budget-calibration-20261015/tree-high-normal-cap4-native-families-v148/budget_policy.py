"""Same grants as the reference policy, with per-plan curves and a heap."""
from collections import Counter
from heapq import heappop, heappush

PRIORITY = {'tight': 3, 'normal': 2, 'loose': 1}
CATEGORY = {'tight': 'coding', 'normal': 'chat', 'loose': 'summarization'}


class TreeBudgetPolicy:
    def __init__(self, model, target_queries=128, max_target_requests=32):
        if target_queries <= max_target_requests:
            raise ValueError('Query limit must also leave room for candidates')
        self.model = model
        self.target_queries = target_queries
        self.max_target_requests = max_target_requests
        self.candidate_roof = target_queries - max_target_requests
        self.counts = Counter()
        self.observed = {}
        self.base = {}
        for cls, category in CATEGORY.items():
            if category not in model['curves']:
                continue
            values = model['curves'][category]
            keys = sorted(map(int, values))
            curve = [0.0]
            for n in range(1, 9):
                lower = max(k for k in keys if k <= n)
                higher = min((k for k in keys if k >= n), default=lower)
                estimate = float(values[str(lower)])
                if higher > lower:
                    estimate += (float(values[str(higher)])-estimate)*(n-lower)/(higher-lower)
                curve.append(estimate)
            self.base[cls] = tuple(curve)

    def gain(self, state, n):
        if n == 0:
            return 0.0
        estimate = self.base[state.slo_class][n]
        rounds, total = self.observed.get((state.slo_class, n), (0, 0))
        estimate = (12*estimate+total)/(12+rounds)
        remaining = max(0, (state.output_token_budget or 256)-state.delivered_tokens)
        return min(remaining, max(1.0, estimate))

    def observe(self, state, n, delivered):
        if 1 <= n <= 8 and state.slo_class in CATEGORY:
            # Intrinsic g(n) must not learn an output-limit-clipped gain.
            # Exclude the whole terminal horizon, independently of whether
            # this particular target verdict happened to reach that limit.
            depth=int(self.model.get('tree_depth',4))
            remaining_before=(state.output_token_budget or 256)-state.delivered_tokens+delivered
            if remaining_before<min(n,depth)+1:
                self.counts[f'terminal_horizon_observations_B{n}']+=1
                return
            key = (state.slo_class, n)
            rounds, total = self.observed.get(key, (0, 0))
            self.observed[key] = (rounds+1, total+delivered)
            self.counts['online_gain_updates'] += 1

    def allocate(self, normal, eager, states, wait_ms, draft_limit=None):
        normal, eager = tuple(normal), tuple(eager)
        indices = normal+eager
        if len(set(indices)) != len(indices):
            raise ValueError('Normal/eager requests overlap')
        left = self.candidate_roof
        if draft_limit is not None:
            left = min(left, max(0, int(draft_limit)))
        grants = dict.fromkeys(indices, 0)
        needs, caps, curves = {}, {}, {}
        for i in indices:
            s = states[i]
            caps[i] = min(8, s.max_gamma or 8)
            curves[i] = [0.0]+[self.gain(s, n) for n in range(1, caps[i]+1)]
            elapsed = max(0.0, s.decode_elapsed_ms-(s.decode_start_elapsed_ms or 0.0))
            delivered = max(0, s.delivered_tokens-1)
            gap = max(0.0, (elapsed+max(0.0, wait_ms))/s.slo_tpot_ms-delivered)
            needs[i] = next((n for n in range(1, caps[i]+1) if curves[i][n] >= gap), None)
            if needs[i] is None:
                self.counts['current_forecast_unfit'] += 1
        order = sorted(indices, key=lambda i: (-PRIORITY[states[i].slo_class], i in eager, i))
        for i in order:
            need = needs[i]
            if need is not None and need <= left:
                grants[i] = need
                left -= need
        for i in order:
            if not grants[i] and i in normal and left:
                grants[i] = 1
                left -= 1
        # An empirical curve can be flat or noisy before a later useful B.
        # Buy the complete smallest positive-gain jump rather than getting
        # stuck at its first zero/negative marginal. Compare gains per actual
        # candidate, retaining request ID as the deterministic tie breaker.
        heap = []
        def push_next(i):
            n = grants[i]
            if 0 < n < caps[i]:
                next_n = next((m for m in range(n+1, caps[i]+1)
                               if curves[i][m] > curves[i][n]), None)
                if next_n is not None:
                    cost = next_n-n
                    score = PRIORITY[states[i].slo_class]*(curves[i][next_n]-curves[i][n])/cost
                    heappush(heap, (-score, i, next_n))
        for i in order:
            push_next(i)
        while left and heap:
            _, i, next_n = heappop(heap)
            cost = next_n-grants[i]
            if cost > left:
                # Remaining capacity only decreases. Do not pay for an
                # incomplete jump whose predicted gain has not increased.
                continue
            grants[i] = next_n
            left -= cost
            push_next(i)
        self.counts['allocation_calls'] += 1
        for n in grants.values():
            self.counts[f'grant_B{n}'] += 1
        assert sum(grants.values()) <= self.candidate_roof
        return ({i: grants[i] for i in normal if grants[i]},
                {i: grants[i] for i in eager if grants[i]})

    def check_target(self, candidate_sizes):
        queries = sum(n+1 for n in candidate_sizes)
        if len(candidate_sizes) > self.max_target_requests or queries > self.target_queries:
            raise RuntimeError('Root-inclusive target verification budget exceeded')
        self.counts['target_calls'] += 1
        self.counts['target_actual_queries'] += queries
        self.counts['target_max_actual_queries'] = max(self.counts['target_max_actual_queries'], queries)
        return queries
