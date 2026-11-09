"""Actual read-only KEEP / DELETE / KEEP regression for a frozen scene.

This checks query mechanics and persistent state, never visual edit correctness.
Frame keys and target tracks come exclusively from supplied data.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path


def check(manifest,queries,frame_keys,output):
    import torch
    from .v44_factory import from_manifest
    scene,m=from_manifest(manifest);queries=json.loads(Path(queries).read_text())
    registered={f.key for f in m.frames}
    if not frame_keys or not set(frame_keys)<=registered:raise ValueError('Explicit registered probe frames required')
    records=[];scene.assert_persistent_unchanged()
    for query in queries:
        ids=m.actor_ids(query['delete_track_ids'])
        if not ids:raise ValueError('Empty deletion regression is not informative')
        visible=False
        for key in frame_keys:
            a=scene.render(key,focus_actor_ids=ids,diagnostic_actor_ids=ids,diagnostic_branch_alpha=True)
            b=scene.render(key,delete_actor_ids=ids,focus_actor_ids=ids,diagnostic_actor_ids=ids,diagnostic_branch_alpha=True)
            c=scene.render(key,focus_actor_ids=ids,diagnostic_actor_ids=ids,diagnostic_branch_alpha=True)
            before=float(a['owner_alpha_T'].abs().max());after=float(b['owner_alpha_T'].abs().max())
            visible|=before>.01
            repeat=float((a['rgb']-c['rgb']).abs().max())
            if repeat!=0 or after>=1e-6 or not torch.equal(a['row_owners'],b['row_owners']):
                raise RuntimeError('Query changed fixed rows or failed mechanical deletion/roundtrip')
            records.append(dict(query=query,frame_key=key,target_alpha_before=before,target_alpha_after=after,
                keep_delete_keep_rgb_maxabs=repeat,immutable_owner_rows=True,coverage=a['coverage'],
                group_conservation_maxabs=max(a['diagnostic_alpha_error_max'],b['diagnostic_alpha_error_max'])))
        if not visible:raise RuntimeError('No visible owned target in the supplied probe frames')
    scene.assert_persistent_unchanged()
    receipt=dict(status='PASS_ACTUAL_FROZEN_SCENE_QUERY_MECHANICS',manifest=str(Path(manifest).resolve()),
        model=str(m.model),iteration=m.iteration,rows=records,parameter_updates=0,
        quality_accepted=None,human_verdict=None,scope='visible owned target, owner removal, immutable rows and exact roundtrip; no hidden-surface correctness claim')
    Path(output).write_text(json.dumps(receipt,indent=2)+'\n');return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['manifest','queries','output']:p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--frames-json',required=True,type=json.loads)
    a=p.parse_args();check(a.manifest,a.queries,a.frames_json,a.output)

if __name__=='__main__':main()
