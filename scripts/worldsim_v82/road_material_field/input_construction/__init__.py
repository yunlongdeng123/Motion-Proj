"""Scene-independent, optional road-inpainting input construction."""
from .case_constructor import construct_case_inputs, make_proposal_reducer
from .input_builder import prepare_native_layers, render_view
from .adaptive_view import plan_views
from .input_utility import choose_view
from .depth_adapter import build_depth_hints, sample_depth_on_grid, reconcile_road_prior
from .lane_structure import build_lane_condition
