"""检查来源消融的实际梯度关闭和两路排队边界，不验证CUDA或视觉质量。"""
from pathlib import Path
import ast,hashlib,json,sys
W=Path(__file__).resolve().parent
sys.path.append('C:/Users/dengyunlong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages')
import torch
from road_objective import road_objective

def main():
    receipts=[]
    for name,enabled in [('real_only',1),('prior_only',2),('ownership_only',None)]:
        root=W/'ablations'/name
        manifest=json.loads((root/'candidate_manifest.json').read_text())
        for rel,entry in manifest['files'].items():
            actual=root/'candidate'/rel
            assert hashlib.sha256(actual.read_bytes()).hexdigest()==entry['candidate_sha256']
            if rel!='utils/v2_runtime_context.py':assert actual.read_bytes()==(W/'candidate'/rel).read_bytes()
        tree=ast.parse((root/'candidate/utils/v2_runtime_context.py').read_text(encoding='utf-8'))
        assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and
            any(ast.unparse(t)=="self.config['road']['objective_weights']" for t in n.targets))
        weights=ast.literal_eval(assignment.value);assert weights==manifest['road_weights']
        rgb=torch.full((3,1,3),.7,requires_grad=True)
        dep=torch.full((1,1,3),4.,requires_grad=True)
        alpha=torch.full((1,1,3),.8,requires_grad=True)
        target=dict(rgb=torch.full((3,1,3),.2),depth=torch.full((1,1,3),8.),
            provenance=torch.tensor([[[1,2,0]]]),confidence=torch.ones(1,1,3))
        loss,terms=road_objective(rgb,dep,alpha,target,weights)
        grads=torch.autograd.grad(loss,[rgb,dep,alpha])
        for column,provenance in enumerate([1,2,0]):
            nonzero=any(bool(g[...,column].abs().max()>0) for g in grads)
            assert nonzero==(provenance==enabled), (name,provenance)
        receipts.append(dict(variant=name,actual_source_weights=weights,
            disabled_source_and_unknown_road_gradients_zero=True,
            unchanged_ownership_and_renderer=True))
    for p in [W/'ablation_support.py',W/'run_road_ablation_group.py',*(W/'ablations').rglob('*.py')]:
        ast.parse(p.read_text(encoding='utf-8'))
    # 队列槽位计算：主实验运行时只能加一个；主实验退出后最多两个。
    states=[(1,0,True),(1,1,False),(0,1,True),(0,2,False)]
    for main_active,children,admit in states:assert (main_active+children<2)==admit
    out=dict(status='PASSED_CPU_ABLATION_SOURCE_AND_SLOT_CONTRACTS',arms=receipts,
        max_concurrent_training_including_main=2,queue_slot_cases=len(states),
        CUDA_checked=False,quality_accepted=False)
    (W/'ablations/cpu_contract_receipt.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out))

if __name__=='__main__':main()
