"""实验入口的阶段检查；当前协议仅允许CPU检查，默认不启动任何任务。"""
from pathlib import Path
import argparse,json

def check_phase(contract,phase,smoke_receipt=None):
    if phase=='inspect':return dict(allowed=True,phase='inspect',GPU_execution=False)
    if contract.get('GPU_authorized') is not True:
        raise PermissionError('当前仅授权CPU准备；GPU smoke与正式实验均未授权。')
    if phase=='formal':
        if not smoke_receipt or smoke_receipt.get('verified') is not True or smoke_receipt.get('native_GPU_executed') is not True:
            raise PermissionError('正式实验须先通过真实GPU生命周期、basis与持久化验收。')
    return dict(allowed=True,phase=phase,GPU_execution='caller must use an explicitly selected execution entry')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--phase',choices=['inspect','smoke','formal'],default='inspect');p.add_argument('--smoke-receipt',type=Path)
    args=p.parse_args();contract=json.loads(args.contract.read_text(encoding='utf-8'))
    receipt=json.loads(args.smoke_receipt.read_text(encoding='utf-8')) if args.smoke_receipt else None
    print(json.dumps(check_phase(contract,args.phase,receipt),ensure_ascii=False))
