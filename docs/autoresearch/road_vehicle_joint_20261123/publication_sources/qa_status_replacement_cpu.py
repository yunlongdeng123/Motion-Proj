"""Check status-tail scope with synthetic cases and a read-only actual origin document."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess
from status_tail_replacement import replace_status_tail,MARKER,LABEL

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True);p.add_argument('--parent',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    replacement='## '+LABEL+'\n\n当前阶段待主审。\n\n历史导航：[原报告](autoresearch/road_vehicle_joint_20261123/README.md)。\n'
    prefix='# 当前研究\n\n## 其它任务\n\n保留原字节。\n\n';old=prefix+MARKER+'\n旧记录\n\n## ROAD-OLD\n另一历史\n'
    new,meta=replace_status_tail(old,replacement);assert new.startswith(prefix) and new==prefix+replacement
    failures=0
    for bad_old,bad_new in [('missing',replacement),(old,replacement+'\n## 第二个状态\n'),(old,'## no task label\n')]:
        try:replace_status_tail(bad_old,bad_new)
        except ValueError:failures+=1
        else:raise AssertionError('An invalid status replacement passed')
    env=dict(os.environ,GIT_OPTIONAL_LOCKS='0');env.pop('GIT_INDEX_FILE',None)
    parent=subprocess.check_output(['git','rev-parse',a.parent+'^{commit}'],cwd=a.repo,env=env).decode().strip()
    raw=subprocess.check_output(['git','show',parent+':docs/RESEARCH_STATUS.md'],cwd=a.repo,env=env);text=raw.decode('utf-8');actual,proof=replace_status_tail(text,replacement)
    start=text.index(MARKER);assert actual[:start].encode('utf-8')==raw[:len(text[:start].encode('utf-8'))]
    out=dict(schema='publication.current_status_replacement_CPU_QA.v1',complete=True,decision='PASS',synthetic_invalid_cases_rejected=failures,
        actual_origin_parent=parent,actual_origin_status_sha256=hashlib.sha256(raw).hexdigest(),actual_origin_scope_check=proof,
        unrelated_prefix_byteexact=True,current_task_sections=1,actual_repo_file_written=False,stage_executed=False,Git_mutations=0)
    assert not a.output.exists();a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
