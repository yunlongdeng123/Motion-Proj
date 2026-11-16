"""准备阶段的硬阻止入口。没有设备导入、后台进程或训练控制器。"""
from pathlib import Path
import argparse
import json

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--prereg',type=Path)
    args=parser.parse_args()
    if args.execute:
        print(json.dumps(dict(executed=False,backend_imported=False,reason='CPU preparation only; wait for explicit user GPU authorization; this entry cannot execute training'),ensure_ascii=False))
        raise SystemExit(3)
    value=json.loads(args.prereg.read_text()) if args.prereg else {}
    print(json.dumps(dict(executed=False,backend_imported=False,execution_enabled=False,
        phase=value.get('status','CPU_PREPARATION_ONLY'),GPU_authorization_required=True),ensure_ascii=False))

if __name__=='__main__':main()
