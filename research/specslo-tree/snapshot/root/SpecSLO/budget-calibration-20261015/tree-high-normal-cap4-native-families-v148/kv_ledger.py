"""Track contextual KV validity when a speculative prefix is overwritten."""


def record_overwrite(state, position, tokens):
    position=int(position);values=list(map(int,tokens))
    if position<int(state['safe_before']) or not values:
        raise RuntimeError('Speculative KV write crosses a protected prefix or is empty')
    # Equal token IDs do not imply equal KV when an ancestor prefix changed.
    # Drop every descendant claim, including physical slots not overwritten
    # by this shorter chain; re-add only queries that actually ran.
    state['tokens']={p:t for p,t in state['tokens'].items() if p<position}
    state['tokens'].update({position+offset:token for offset,token in enumerate(values)})
