import ast, datetime, difflib, hashlib, json
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
source=HERE/'loss_audit/source_frozen/train.py'
candidate=HERE/'reset_candidate'
candidate.mkdir(exist_ok=True)
old=source.read_text(encoding='utf8')
needle='    if (iteration % args.opacity_reset_interval == 0 or (\n            args.white_background and iteration == args.densify_from_iter)) and not dynamic:'
replacement='    if (iteration < args.densify_until_iter and not dynamic and\n            (iteration % args.opacity_reset_interval == 0 or\n             (args.white_background and iteration == args.densify_from_iter))):'
assert old.count(needle)==1
new=old.replace(needle,replacement)
compile(new,'candidate/train.py','exec')
(candidate/'train.py').write_text(new,encoding='utf8')
(candidate/'stop_late_static_reset.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/train.py',tofile='b/train.py')),encoding='utf8')

# Execute the actual frozen/candidate helper at the point-count cap: density
# updates must be skipped, while its reset scheduling still executes.
def load_function(text):
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='densification_and_optimization')
    ns={}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual helper>','exec'),ns)
    return ns[node.name]
class Model:
    get_xyz=SimpleNamespace(shape=(3000000,3))
    def __init__(self):self.resets=0
    def reset_opacity(self):self.resets+=1
args=SimpleNamespace(dynamic_densify_until_iter=15000,densify_until_iter=15000,
    densify_until_num_points=3000000,dynamic_densify_until_num_points=1000000,
    opacity_reset_interval=3000,white_background=False,densify_from_iter=500)
rows=[]
for white in [False,True]:
    args.white_background=white
    for dynamic in [False,True]:
        for iteration in [499,500,501,2999,3000,3001,12000,14999,15000,15001,18000,30000,33000,48000,50000]:
            counts=[]
            for text in [old,new]:
                m=Model();load_function(text)(m,args,iteration,None,None,None,None,dynamic);counts.append(m.resets)
            if iteration<15000:assert counts[0]==counts[1]
            if dynamic:assert counts==[0,0]
            if iteration>=15000:assert counts[1]==0
            rows.append(dict(iteration=iteration,dynamic=dynamic,white_background=white,baseline_resets=counts[0],candidate_resets=counts[1]))
result={'status':'CPU schedule checks pass; not deployed and not GPU validated',
 'type':'isolated training schedule correction candidate, not a precision-preserving infra change',
 'rationale':'Static opacity reset continues after topology updates end; exact 30k checkpoint is saved immediately after the reset.',
 'paper_claim':'Paper does not specify reset schedule. This candidate needs controlled effect verification, and cannot alone fix existing branch leakage.',
 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'candidate_sha256':hashlib.sha256((candidate/'train.py').read_bytes()).hexdigest(),
 'changes':'Add iteration<densify_until_iter to static reset condition; all losses and earlier reset behavior unchanged.',
 'tests':rows,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(candidate/'candidate_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'candidate':str(candidate),'tests':len(rows),'deployed':False}))
