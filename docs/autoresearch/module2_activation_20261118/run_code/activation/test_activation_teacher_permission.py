"""源许可审计的CPU行为测试；不访问native、CUDA或冻结数据。"""
import numpy as np
from prepare_activation_supervision import corner_values, source_roles, choose_source


def test_corner_convention_uses_four_native_neighbors():
    image = np.arange(20).reshape(4, 5)
    got = corner_values(image, np.array([[1.5, 2.5], [.5, .5]]))
    np.testing.assert_array_equal(got, [[11, 12, 16, 17], [0, 1, 5, 6]])


def test_K_and_source_boundary_veto_all_four_neighbors():
    n = 3
    yes = np.ones((n, 4), bool)
    no = np.zeros((n, 4), bool)
    W, K = yes.copy(), no.copy()
    W[1, 3] = False
    K[2, 0] = True
    got = source_roles(np.ones(n, bool), np.ones(n), np.ones(n), W, K, no, no, yes, no)
    np.testing.assert_array_equal(got['S_permission'], [True, False, False])
    # Zero bilinear weights cannot authorize a retained neighbor.
    assert got['K_any4'][2]


def test_inherited_view_gates_are_strict_without_old_request():
    yes = np.ones((4, 4), bool)
    no = np.zeros((4, 4), bool)
    got = source_roles(np.ones(4, bool), np.array([59.9, 60., 1., 1.]),
        np.array([.026, .1, .025, .024]), yes, no, no, no, yes, no)
    np.testing.assert_array_equal(got['authorized'], [True, False, False, False])
    # No requested/legal_write/current-alpha argument exists in fresh qualification.


def test_source_selection_prefers_real_A_and_never_averages_pseudo():
    valid = np.array([[True, True, False], [True, True, False]])
    role = np.array([[1, 3, 0], [3, 3, 0]])
    rgb = np.array([[[.1,.2,.3],[.2,.3,.4],[0,0,0]],
                    [[.9,.8,.7],[.8,.7,.6],[1,1,1]]])
    score = np.array([[1,1,1],[10,10,10]])
    selected, permitted, color = choose_source(valid, role, rgb, score)
    np.testing.assert_array_equal(selected, [0,1,0])
    np.testing.assert_array_equal(permitted, [True,True,False])
    np.testing.assert_array_equal(color[:2], [rgb[0,0],rgb[1,1]])
    assert np.isnan(color[2]).all()
