"""One-time independent auditor projection; no actual read or runner execution."""
import ast,hashlib,io,re,tokenize
from pathlib import Path
D=Path('/root/SpecSLO/budget-calibration-20261015/budget-diagnostics');old=D/'audit_v205_terminal_HOST.py';out=D/'audit_v209_terminal_HOST.py'
assert hashlib.sha256(old.read_bytes()).hexdigest()=='2ab9b0e074fb8b35f739c5aaeeee0921ef92ee7655fe52ed881d8de67b1607c8'
assert not out.exists()
text=old.read_text();lines=text.splitlines(keepends=True);offsets=[0]
for line in lines:offsets.append(offsets[-1]+len(line))
protected=[t for t in tokenize.generate_tokens(io.StringIO(text).readline)if t.type==tokenize.NUMBER or t.type==tokenize.STRING and re.fullmatch(r"['\"][0-9a-f]{64}['\"]",t.string)]
for t in reversed(protected):
 a=offsets[t.start[0]-1]+t.start[1];b=offsets[t.end[0]-1]+t.end[1];text=text[:a]+text[a:b].replace('205','\x00E205\x00')+text[b:]
text=text.replace('205','209').replace('\x00E209\x00','205')
def replace(old,new):
 global text
 assert text.count(old)==1,(old,text.count(old));text=text.replace(old,new)
replace('tree-causal80-small-cohort-single-writer-full-v209','tree-causal80-small-cohort-role-aware-audit-full-v209')
replace('V209_SINGLE_WRITER_SEVEN_N_FROM_FROZEN202_SOURCE_AND_GUARD_READY_FOR_ROOT_REVIEW.json','V209_ROLE_AWARE_RETAINED_AUDIT_SOURCE_AND_GUARD_READY_FOR_ROOT_REVIEW.json')
replace("CPU_SHA='a560a56c807c3600e231e230b80869768731c92772e47f707f4b7fb2520b3c78'","CPU_SHA='8c1c085cb831ff26a28fda0ba92e908cd730bfdf067b09c25304a5708f5e60d0'")
replace("str(READY):'737be8d457bab4228acd3d1dfe473f34b8c8fa6632967712d9789a9a7d16d1b0'","str(READY):'6b4a8e617cc128cae7ae98a72730ecefc19e3c2af03e2fe71cf1ffd5656a52d6',\n str(D/'audit_v205_terminal_HOST.py'):'2ab9b0e074fb8b35f739c5aaeeee0921ef92ee7655fe52ed881d8de67b1607c8'")
replace("FINAL_FREEZE_SHA='3fbcdad8b794d323027d7685fbb73ac7e356420107f7309af8540fb134fed01a'","FINAL_FREEZE_SHA=None # ROOT final exact freeze not yet supplied; CLI hard-disabled")
replace("FINAL_ARCHIVE_SHA='d4fe73947ca82133cab3f2b0802b90620b195014ceb222bd6c3626f4f5d48031'","FINAL_ARCHIVE_SHA=None")
replace("FINAL_ROOT_PREPARED_SHA='ef8c2077c2820717aa4191bf166a624a8c95def1ff1f57dd975bda1a4f5b0484'","FINAL_ROOT_PREPARED_SHA=None")
replace("FINAL_PEER_SHA='67794d6c4240cad58d74074314cf901ad2003a93b1ee3db54e5a978ff2dd'","FINAL_PEER_SHA='e415d39acd3c9cfe65c624771b9d1b126f0c95d06adaf01616fb8e45c6829ba2'") if "FINAL_PEER_SHA='67794d6c4240cad58d74074314cf901ad2003a93b1ee3db54e5a978ff2dd'" in text else replace("FINAL_PEER_SHA='67794d6c4240cad58d74074314cf901ad2003a93b1ee3ffe9db54e5a978ff2dd'","FINAL_PEER_SHA='e415d39acd3c9cfe65c624771b9d1b126f0c95d06adaf01616fb8e45c6829ba2'")
replace("c['tests_run']==7","c['tests_run']==3")
replace('actual current209 CPU7 producer schema','actual current209 CPU3 producer schema; donor205 seven not counted')
replace("len(pr['actual_HOST_attempts'])==3 and len(pr['retained_HOST_files'])==170,'three actual HOST attempts/all170 source-log-result files'","len(pr['actual_HOST_attempts'])==3 and len(pr['retained_HOST_files'])==96,'three own209 HOST attempts/all96 source-log-result files'")
replace("need(len(pins)==968 and len(evidence)==174 and len(f['files'])==26 and len(f['external_files'])==760", "need(len(evidence)==100 and len(f['files'])==26 and len(f['external_files'])==1395")
replace("need(len(reviewed)==966 and len(reviewed)-len(f['peer_reviews'])==965,'actual965 beforepeer,966 ROOT flat review')", "need(len(reviewed)==len(pins)-2 and len(reviewed)-len(f['peer_reviews'])==1526,'actual current209 flat source closure; archive count derived only from future frozen map')")
replace("auxiliary=root['auxiliary_ready_evidence'];need(len(auxiliary)==3 and root['auxiliary_ready_evidence_is_outside_guard_archive']is True", "auxiliary=root['auxiliary_ready_evidence'];need(bool(auxiliary)and root['auxiliary_ready_evidence_is_outside_guard_archive']is True")
replace("'three separate actual cold evidence inputs, not added tar members'", "'explicit ROOT-reviewed separate actual cold/source evidence, not invented archive members'")
replace("rp['actual_source_members']==rp['actual_archive_members']==968", "rp['actual_source_members']==rp['actual_archive_members']==len(pins)")
replace("from base209 import require_source_parent\n donor=require_source_parent(archive=True)","from base209 import require_source_parent,require_derivation205\n donor205=require_derivation205(archive=True)\n need(donor205['members']==968 and donor205['archive_verified_in_place']is True and donor205['production_parent_remains197']is True and donor205['device_qualification_inherited']is False,'exact frozen205968 source donor only; failed old model run is not current qualification')\n donor=require_source_parent(archive=True)")
replace("current_HOST_test7_scope='real4process shared filesystem and actual actor lambda; original stage4 and Native NPU fence NOT executed'", "current_HOST_test3_scope='three new source/retained-role HOST checks; old205 seven verified donor-only and no newengine math',source_donor205=donor205")
replace("dp=row['draft'];need(dp['raw_encoded_shape']", "dp=row['draft'];need(type(dp)is dict and dp['variable_num_draft_tokens']==[4]*16 and dp['raw_encoded_shape']")
replace("    else:\n     target,reference,old,x=", "    else:\n     need(row['draft']is None and all(type(row[k])is dict for k in('target','reference','old_control','crossM')),'target retains every own proof and no fabricated draft proof')\n     target,reference,old,x=")
replace("dp=nxt['draft_proof'];need(dp['raw_encoded_shape']==[16,4,2]and all(dp[k]is True for k in('all_physical_hidden_original_matcher','all28_query_KV_original_matcher','exact_full8_encoded')),'ownnext original all56KV/full8 proof retained')", "dp=nxt['draft_proof'];need(type(dp)is dict and dp['variable_num_draft_tokens']==[4]*16 and dp['raw_encoded_shape']==[16,4,2]and all(dp[k]is True for k in('all_physical_hidden_original_matcher','all28_query_KV_original_matcher','exact_full8_encoded'))and dp['source_KV_copied']is False,'ownnext original all56KV/full8 proof retained')\n   else:need(nxt['draft_proof']is None,'target ownnext retains no fabricated draft proof')")
replace(" receipt_path=P/('receipt-'+args.tag+'.json')", " need(all(type(v)is str and re.fullmatch(r'[0-9a-f]{64}',v)is not None for v in(FINAL_FREEZE_SHA,FINAL_ARCHIVE_SHA,FINAL_ROOT_PREPARED_SHA)),'Future209 terminal CLI remains disabled until exact ROOT final freeze/archive/prepared pins are bound in a reviewed revision')\n receipt_path=P/('receipt-'+args.tag+'.json')")
replace("'Failure stays failure, no full GP or adoption is assigned. Final archive member count is computed", "'Only actual own3 HOST is current source evidence; failed old205 actual/raw fixture is never current209 numerical qualification. Failure stays failure, no full GP or adoption is assigned. Final archive member count is computed")
ast.parse(text);out.write_text(text)
print(hashlib.sha256(out.read_bytes()).hexdigest())
