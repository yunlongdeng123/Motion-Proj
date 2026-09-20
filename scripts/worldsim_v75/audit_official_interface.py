"""CPU 审计官方逐场景输入绑定，不加载模型或代替推理。"""
from __future__ import annotations
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def audit(source: Path):
    source = source.resolve()
    base = source / 'integrations_v2/omnidreams/impl/eval'
    # 只导入两个标准库模块；命名空间路径指向原文件，函数和数据类型均未改写。
    for name, relative in [('omnidreams','integrations_v2/omnidreams'),
                           ('omnidreams.impl','integrations_v2/omnidreams/impl'),
                           ('omnidreams.impl.eval','integrations_v2/omnidreams/impl/eval')]:
        mod = types.ModuleType(name)
        mod.__path__ = [str(source / relative)]
        sys.modules[name] = mod
    m = load_file('omnidreams.impl.eval.manifest', base/'manifest.py')
    g = load_file('omnidreams.impl.eval.generation', base/'generation.py')
    cases=[]
    for name in ['case_A','case_B']:
        case=m.EvalCase(uuid=name,camera=m.DEFAULT_CAMERA,dataset_repo='fixture/contract-only',
                       dataset_revision='not-a-real-dataset',dataset_subpath='',
                       reference_video=m.AssetRef(name+'/rgb.mp4'),hdmap_video=m.AssetRef(name+'/map.mp4'),
                       prompt=m.AssetRef(name+'/prompt.txt'))
        p=Path('/contract-only')/name
        staged=m.StagedCase(case=case,reference_video_path=p/'rgb.mp4',hdmap_video_path=p/'map.mp4',
                            prompt_path=p/'prompt.txt',first_frame_path=p/'first.png',prompt_text='Identical control prompt')
        result=g.generation_result_for_case(staged,run_root=Path('/contract-output'),
                    recipe='interactive-drive-omnidreams',total_blocks=16,flashdreams_run='flashdreams-run')
        command=list(result.command)
        model_command=[]
        i=0
        while i<len(command):
            if command[i] in ('--output-path','--stats-path'):
                i+=2
            else:
                model_command.append(command[i]);i+=1
        cases.append({'case':name,'first_frame':str(staged.first_frame_path),'hdmap':str(staged.hdmap_video_path),
                      'command':command,'model_command':model_command,
                      'first_frame_bound':str(staged.first_frame_path) in command,
                      'hdmap_bound':str(staged.hdmap_video_path) in command})
    nodes=ast.parse((base/'generation.py').read_text(encoding='utf-8'))
    function=next(x for x in nodes.body if isinstance(x,ast.FunctionDef) and x.name=='generation_result_for_case')
    # 检查全函数，确保没有遗漏 side effect 或间接分支。
    forbidden=['first_frame_path','hdmap_video_path','reference_video_path']
    used=sorted({x.attr for x in ast.walk(function) if isinstance(x,ast.Attribute) and x.attr in forbidden})
    commit=subprocess.run(['git','-C',str(source),'rev-parse','HEAD'],check=True,text=True,capture_output=True).stdout.strip()
    return {'task_id':'WS-V75-QUALIFY-01','run_id':'20260920-r1','source_commit':commit,
            'scope':'CPU execution of official command builder; zero model forwards',
            'binding':{'cases':cases,'model_commands_equal':cases[0]['model_command']==cases[1]['model_command'],
                       'source_input_path_attributes_used':used},
            'decision':'Do not use the audited high-level batch generator until scene binding is fixed. Use explicit pipeline inputs.',
            'failure_ledger_refs':['V74-H2-F20','V74-H2-F21','V74-H2-F22'],
            'failure_ledger_delta':'V75-F01: engineering input binding; no scientific failure inferred','human_verdict':None}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=audit(a.source)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'input_binding':result['binding']['model_commands_equal'],
                     'path_attributes_used':result['binding']['source_input_path_attributes_used'],
                     'model_forwards':0},ensure_ascii=False))
