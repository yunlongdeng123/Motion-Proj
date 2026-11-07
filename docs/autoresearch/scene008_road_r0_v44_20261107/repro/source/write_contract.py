"""Freeze the R0 mechanism contract before paired extension scores are read."""
from pathlib import Path
import hashlib
import json

RUN = Path(__file__).resolve().parents[1]
target = RUN/'contract.json'
contract = dict(
    schema='SCENE008_R0_V44_MEASURED_EXTENSION_V1', date='2026-11-07',
    vehicle_asset_sha256='0b1f69e13d0e54328734325c8f118bc27dc7b75080869f2d52f55038afc0b20d',
    vehicle_posterior_sha256='b98a0d932730fbf1462e555eb7d5eb931030b51fe68848990f073b9752879e06',
    measured_surface_sha256='1f3fc2f52280d984a27fe7684021f2040205deba8badac878eb614401879a767',
    region='stable real-source supported cells forward of the frozen V44 road maximum local-x',
    variants=['frozen V44', 'same V44 plus always-active measured road extension',
              'optional same extension plus separately registered persistent old-row retirement'],
    geometry_sigma_m=[.052,.052,.015], peak_opacity=.99,
    camera_and_edit_policy='original native camera, trajectory, single depth sort; delete changes only a37 existence',
    appearance='real FRONT199 stable radiance cluster only; no generated target; no new residual field',
    intervention='world-asset addition, no image paste or edited-state bypass',
    no_optimizer=True,
    mechanism_acceptance=dict(verified_unoccluded_road_solo_alpha_at_least=.98,
        verified_unoccluded_environment_T_at_most=.02,
        require_measured_depth_and_source_agreement=True,
        note='Alpha thresholds describe supported unoccluded targets, not a completion accuracy metric'),
    preservation_gates=dict(
        a6_RGB_MAE_max_frame_regression=.001,
        a37_keep_RGB_MAE_max_frame_regression=.001,
        a6_measured_depth_MAE_max_frame_regression_m=.01,
        original_observed_road_RGB_pooled_MAE_max_regression=.001,
        protected_RGB_pooled_MAE_max_regression=.001,
        original_observed_road_RGB_max_frame_regression=.005,
        protected_RGB_max_frame_regression=.005,
        rule='All gates required; gains on new road cannot compensate for foreground failure; report tails and changes separately'),
    benchmark='FRONT-only source scope; same-scene mechanism experiment, not unseen-scene or never-visible validation',
    power='KEEP_ON',
)
if target.exists():
    raise FileExistsError(target)
target.write_text(json.dumps(contract, indent=2, ensure_ascii=False)+'\n')
print(str(target), hashlib.sha256(target.read_bytes()).hexdigest())
