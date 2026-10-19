"""Copy only actual reviewed core sources/configs and review guides into Git."""
import json
from publish_ablation_nodes import W,O,REPO,DOC,CODE,copy,write,long
from build_ai_review_bundle import D,S

def main():
    target=CODE/'ai_review_sources';paths=[]
    for p in sorted((D/'code').rglob('*')):
        if p.is_file():
            rel=target/'code'/p.relative_to(D/'code');copy(p,REPO/rel);paths.append(rel.as_posix())
    # Actual source bytes and upstream license hashes must survive Git filters.
    attributes='code/** -text\n# Preserve immutable source and license bytes, including upstream whitespace.\n'
    attributes+=''.join('code/'+folder+'/** -whitespace\n' for folder in ['base','overlays','entrypoints','binding_tools','review_tools','licenses','diffs'])
    rel=target/'.gitattributes';write(REPO/rel,attributes);paths.append(rel.as_posix())
    for arm in ['baseline','full','real_only','prior_only','ownership_only']:
        for name in ['effective_config.json','status.json','deployed_manifest.json','v2_run_contract.json']:
            p=S/'runs'/arm/name
            if p.exists():
                rel=target/'run_metadata'/arm/name;copy(p,REPO/rel);paths.append(rel.as_posix())
    for name in ['source_snapshot_manifest.json','entrypoints_manifest.json','additional_sources_manifest.json']:
        rel=target/name;copy(D/'evidence'/name,REPO/rel);paths.append(rel.as_posix())
    for name in ['collect_ai_review_sources.py','build_ai_review_bundle.py','verify_ai_review_bundle.py','publish_ai_review_sources.py']:
        rel=CODE/name;copy(W/name,REPO/rel);paths.append(rel.as_posix())
    guide=(D/'README.md').read_text(encoding='utf-8')
    guide=guide.replace('- Code commit: `67d57b88a10a1340c0e6a4537808f8f4324f2c4c`','- 代码快照与本指南同提交发布；最终ZIP的PROVENANCE.json记录确切代码提交。')
    for name in ['README.md','REVIEW_TASK.md','experiments.json','architecture.svg']:
        rel=DOC/'ai_review_bundle'/name
        long((REPO/rel).parent).mkdir(parents=True,exist_ok=True)
        if name=='README.md':write(REPO/rel,guide)
        else:copy(D/name,REPO/rel)
        paths.append(rel.as_posix())
    rel=target/'README.md'
    write(REPO/rel,'# Actual code snapshot for AI review\n\nNative baseline core is stored once under code/base. Apply one of code/overlays/{full,real_only,prior_only,ownership_only} to obtain an exact current engine core; code/materialize.py only copies and verifies files. code/source_map.json contains all per-experiment actual hashes. Baseline has no new ownership/real-road/prior-road modules. Actual configs/statuses are under run_metadata; original absolute paths are retained as provenance. No weights, datasets or CUDA binaries are committed here.\n')
    paths.append(rel.as_posix())
    (W/'publish_ai_review_sources_paths.json').write_text(json.dumps(paths,indent=2),encoding='utf-8')
    print(json.dumps({'explicit_paths':len(paths),'target':target.as_posix()}))

if __name__=='__main__':main()
