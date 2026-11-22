"""Build a review-only F06 candidate from origin and actually rebuild its navigation."""
from pathlib import Path
import argparse,os,subprocess,json
from origin_failure_index_cpu import CARD,main as unused,rebuild

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True);p.add_argument('--parent',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    env=dict(os.environ,GIT_OPTIONAL_LOCKS='0');env.pop('GIT_INDEX_FILE',None)
    def git(*args):return subprocess.check_output(['git',*args],cwd=a.repo,env=env)
    parent=git('rev-parse',a.parent+'^{commit}').decode().strip();original=git('show',parent+':'+CARD)
    a.output.mkdir(parents=True,exist_ok=False)
    addition='\n\n## ROAD-VEHICLE-JOINT-20261123：出版接线候选，待主审\n\n此段仅供临时快照重建 failure 导航的 CPU 预检，不是最终发布叙述。R8 原正式 400 步、持久资产与 fresh 验证保留为历史对照；当前车辆沿用原 A20 r3，整场景结论由实际新阶段及主审另填。\n\nR10 新 17 帧几何与来源分别验收。75 个旧同位置锚因新 XYZ 位移重新资格核验；新分支保留 6332 个未受影响锚，当前 17 帧全域保守资格未新增锚，不等于证明所有照片不可见。所有旧版本和失败记录保留，人工 verdict 未填写。\n'
    candidate=a.output/'V82-F06.candidate.md';candidate.write_bytes(original.rstrip()+addition.encode('utf-8'))
    subprocess.check_call([__import__('sys').executable,str(Path(__file__).with_name('origin_failure_index_cpu.py')),'--repo',str(a.repo),'--parent',parent,'--candidate-v82',str(candidate),'--output',str(a.output/'actual_rebuild')])
    (a.output/'preflight_scope.json').write_text(json.dumps(dict(complete=True,review_only=True,finalized_by_root=False,source_parent=parent,candidate=str(candidate),actual_builder_run=True,stage_executed=False,Git_mutations=0),indent=2)+'\n')
if __name__=='__main__':main()
