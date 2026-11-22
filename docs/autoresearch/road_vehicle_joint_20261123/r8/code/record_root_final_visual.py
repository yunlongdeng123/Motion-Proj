"""Primary visual verdict for the completed constrained native experiment."""
from pathlib import Path
import datetime,hashlib,json

ROOT=Path(__file__).resolve().parents[1]
def pin(p):return dict(path=p.as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
summary=ROOT/'formal400_summary_v1/compact_result.json'
result=json.loads(summary.read_text(encoding='utf-8'))
assert result['complete'] and result['all_fresh_and_independent_evidence_ready']
assert result['source_sha256']=='2eaf1569a97fb8ee6907696dfa28db91bb5f3072f74d37491007c87012e2c647'
frames=[86,110,134,179,198]
review=dict(schema='R8.root_formal400_visual_review.v1',complete=True,
    recorded_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    decision='NOT_ACCEPTED_AS_WHOLE_SCENE_RESTORATION',
    shared_input_colour_seam_accepted=True,engineering_asset_and_native_reload_PASS=True,
    native_formal400_complete=True,whole_scene_restoration_accepted=False,
    publish_as_experimental_comparison=True,promote_over_selected_A20_baseline=False,
    shared_source_sha256=result['source_sha256'],numeric_evidence=pin(summary),
    reviewed_original_resolution_shared_images=[pin(ROOT/f'formal400_preview_pngs/{raw:03d}_shared.png') for raw in frames],
    per_view=[dict(raw=raw,resolution=[1152,768],whole_scene_visual='NOT_ACCEPTED',
        observed=['Road interior has more even tone, with persistent broken pavement/edge structure.',
                  'Red and blue-grey ground-level remnants below the pickup remain visible.']) for raw in frames],
    findings=[
        'The source colour/seam PASS is retained. Root reviewed every original-resolution final shared Gaussian image separately from the shared-surface preview.',
        'Paired fixed-scope measurements show road target-colour and observed vehicle RGB improvement. These are not proof of whole-scene reconstruction or unseen road ground truth.',
        'Visible residuals prevent promotion as a completed restoration. The canonical actor-only and old-context influence layers must be inspected before assigning a residual to a particular vehicle.',
        'The historical teacher rim domain contains1792 pixels; actual native renders have only2 dark-threshold pixels there in A0, atlas0 and final. These counts are not interchangeable.',
        'Scene alpha is sufficient in the tested target domains. No scale/opacity expansion was used to hide colour or ownership errors.'
    ],
    constraints=dict(all_centers_geometry_sigma_opacity_normals_tracks_fixed=True,all_five_K_RGB_exact=True,
                     new_diffusion_calls=0),
    root_review_nativeF=0,root_review_nativeB=0,root_review_updates=0,
    human_verdict=None,problem_solved=None)
out=ROOT/'formal400_summary_v1/root_final_visual_review.json';assert not out.exists()
out.write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(pin(out)))
