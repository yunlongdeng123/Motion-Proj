"""离线 CPU 核对权重文件与推理模块导入；不实例化网络。"""
import importlib
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
RUN = Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-HYBRID-BG-20260927/r1')
SOURCE = Path('/root/autodl-tmp/third_party/worldsim_v77/DiffuEraser')


def main():
    from safetensors import safe_open
    manifest = json.loads((RUN/'download_manifest.json').read_text())
    rows = []
    for entry in manifest['files']:
        path = Path(entry['local_path'])
        assert path.is_file(), path
        assert path.stat().st_size == entry['expected_bytes'], path
        row = dict(path=str(path), bytes=path.stat().st_size, size_ok=True)
        if path.suffix == '.json':
            assert isinstance(json.loads(path.read_text()), dict)
            row['json_readable'] = True
        elif path.suffix == '.safetensors':
            with safe_open(str(path), framework='pt', device='cpu') as source:
                keys = list(source.keys())
                assert keys
                for key in keys:
                    source.get_slice(key).get_shape()
                row['tensor_metadata_count'] = len(keys)
                row['tensor_metadata_readable'] = True
        rows.append(row)
    for entry in manifest['reused']:
        path = Path(entry['path'])
        assert path.is_file() and path.stat().st_size == entry['bytes']
    sys.path.insert(0, str(SOURCE))
    modules = {}
    for name in ['torch','torchvision','diffusers','transformers','accelerate','peft','av',
                 'diffueraser.diffueraser','propainter.inference']:
        mod = importlib.import_module(name)
        modules[name] = getattr(mod, '__version__', 'imported')
    from transformers import CLIPTokenizer
    root = Path(manifest['files'][0]['local_path']).parents[2]
    # 独立验证 tokenizer 可本地加载，不做 text encoder 前向。
    tokenizer = CLIPTokenizer.from_pretrained(str(root/'stable-diffusion-v1-5/tokenizer'), local_files_only=True)
    assert tokenizer.vocab_size > 0
    result = dict(state='weights_and_imports_validated_waiting_user_gpu', files=rows,
                  reused=manifest['reused'], modules=modules, tokenizer_vocab=tokenizer.vocab_size,
                  new_weight_bytes=sum(r['bytes'] for r in rows),
                  model_instantiated=False, forward_count=0, gpu_inference_started=False,
                  gpu_compatibility_verified=False, auto_resume=False,
                  validation_scope='file sizes, JSON, safetensors metadata, offline imports/tokenizer; no tensor-value scan or model forward',
                  human_verdict=None)
    (RUN/'preparation_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['files','reused']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
