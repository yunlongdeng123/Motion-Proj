"""不加载网络的checkpoint与数据清单检查。"""
import argparse,json,math
from pathlib import Path
from safetensors import safe_open
def read(p):return json.loads(p.read_text())
def selected_key(k):
    return k.startswith('model.diffusion_model.') and '_3d' not in k and '.transformer_blocks.' in k and '.attn1.' in k and any(s in k for s in ['.to_q.','.to_k.','.to_v.','.to_out.'])
def main(root,task,official):
    checkpoint=(official/'checkpoints/model.safetensors').resolve();weights=[];total=0
    with safe_open(str(checkpoint),framework='pt',device='cpu') as f:
        for key in f.keys():
            shape=f.get_slice(key).get_shape();numel=math.prod(shape);total+=numel
            if selected_key(key):weights.append({'name':key,'shape':shape,'numel':numel})
    old=read(task/'r2/training_admission.json');r3=read(task/'r3/review/review_manifest.json')
    result={'checkpoint':str(checkpoint),'checkpoint_bytes':checkpoint.stat().st_size,'checkpoint_parameter_elements':total,
            'selected_trainable_tensors':weights,'selected_numel':sum(r['numel'] for r in weights),
            'selection':'existing non-3D spatial transformer self-attention Q/K/V/out; no new network modules',
            'r2_technical_candidates':[r['case_id'] for r in old['cases'] if r['training_admission']=='train_usable_pending_human'],
            'r3_keys':list(r3),'r3_case_count':len(r3['clips'])}
    (root/'training_inspection.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('TRAIN_INSPECT',len(weights),'tensors',result['selected_numel'],'elements; r2',len(result['r2_technical_candidates']),'r3',result['r3_case_count'])
    print('EXAMPLE_KEYS',[r['name'] for r in weights[:6]])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--task',type=Path,required=True);p.add_argument('--official',type=Path,required=True);a=p.parse_args();main(a.root,a.task,a.official)
