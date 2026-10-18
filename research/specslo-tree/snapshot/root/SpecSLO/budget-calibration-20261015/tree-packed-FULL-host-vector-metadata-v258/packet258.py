"""Same FULL metadata, one transfer for integer fields plus the original mask.

No model tensors or weights change. Every original HOST validation remains.
Dynamic inputs, positions, slots, request IDs and masks are fresh each call.
This prototype is not qualified for neural or serving performance use.
"""
import ast,copy,hashlib,inspect
from pathlib import Path

def prepared(host):
    source=Path(host.__file__).read_text()
    if hashlib.sha256(source.encode()).hexdigest()!='5d5a09c963ded98396378ddc7d5bfad81736bf49102c9507ad8d3ff9dd77eb44':
        raise RuntimeError('exact original frozen HOST metadata source required')
    original=copy.deepcopy(next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='make_host_tree_attention_metadata'))
    node=copy.deepcopy(original)
    locations=[i for i,n in enumerate(node.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='slots'for t in n.targets)]
    if len(locations)!=1:raise RuntimeError('one original fully checked HOST slot assignment')
    added=ast.parse('''
if _uses_fused_attention(device, attention) and not all_linear:
    return _packet258_full(engine, inputs, logical_positions, physical_positions, request_ids, slots,
                           physical_tables, query_lengths, sequence_lens, masks)
''').body[0]
    node.body.insert(locations[0]+1,added)
    inverse=copy.deepcopy(node);inverse.body.pop(locations[0]+1)
    if ast.dump(inverse,include_attributes=False)!=ast.dump(original,include_attributes=False):
        raise RuntimeError('complete unchanged original validation and fallback AST inverse')
    def pack(engine,inputs,logical,physical,request_ids,slots,physical_tables,query_lengths,sequence_lens,masks):
        import torch
        # Align all integer tensor views to512bytes, including the slot input
        # subsequently cast to the original int32 dtype on the same device.
        values=[];layout={}
        for name,items in [('inputs',inputs),('positions',logical),('requests',request_ids),('slots',slots)]:
            offset=(len(values)+63)//64*64
            values.extend([0]*(offset-len(values)));layout[name]=(offset,len(items));values.extend(items)
        device=engine.model.embed_tokens.weight.device
        packet=torch.tensor(values,dtype=torch.long,device='cpu').to(device=device)
        views={name:packet.narrow(0,offset,count)for name,(offset,count)in layout.items()}
        request_tables=physical_tables.index_select(0,views['requests'])
        cumulative=[]
        for length in query_lengths:cumulative.append((cumulative[-1]if cumulative else 0)+length)
        metadata=host.NativeAttentionMetadata(
            slot_mapping=views['slots'].to(dtype=torch.int32),
            context_lens=torch.tensor([position+1 for position in physical],dtype=torch.int32),
            block_tables=request_tables,actual_seq_lengths_q=tuple(cumulative),sequence_lens=tuple(sequence_lens),
            request_block_tables=request_tables,attention_mask=None,use_fused_infer_attention=True,
            tree_attention=True,tree_attention_mask=host.make_tree_fia_mask(masks).to(device=device,dtype=torch.bool))
        engine._packet258_HOST_calls=getattr(engine,'_packet258_HOST_calls',0)+1
        engine._packet258_last_layout=dict(fields=layout,integer_payload_bytes=len(values)*8,
            dynamic_storage_reused=False,nominal_integer_transfers=1,nominal_mask_transfers=1)
        return views['inputs'],views['positions'],metadata
    namespace=dict(vars(host),_packet258_full=pack)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),__file__+'::original-validated-FULL-packet-projection','exec'),namespace)
    function=namespace['make_host_tree_attention_metadata']
    if inspect.signature(function)!=inspect.signature(host.make_host_tree_attention_metadata):
        raise RuntimeError('preserve all original positional and keyword-only parameters')
    return function,dict(original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        entire_original_validation_and_fallback_AST_inverse=True,original_default_signature=True,
        mathematical_metadata_unchanged_intended=True,dynamic_tensor_data_cached=False,
        neural_qualified=False,performance_qualified=False)
