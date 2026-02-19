# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for construction of jump oeprators.

The tests cover:
  - Correct spectral decomposition of a system-operator in a Bose Hubbard model.

These tests ensure that the form of the returned jump operators is valid.
"""

# ignore non-lowercase variable names for physics notation
# ruff: noqa: N806


import numpy as np

from mqt.yaqs.analog.jump_construction import jump_operator_decomposition


def test_jump_ops_finiteness() -> None:
    """
    Test that jump_operators_from_L produces finite, well-formed operators.

    This test creates a random Hermitian Hamiltonian H and operator L, then calls
    jump_operators_from_L with reasonable physical parameters. It checks that each
    returned jump operator contains no NaNs or infinities, and that its Frobenius
    norm is finite. This ensures numerical stability of the construction.
    """
    N = 8

    rng = np.random.default_rng(12345)

    A = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    H = (A + A.conj().T) / 2

    B = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    L = (B + B.conj().T) / 2

    c_ops = jump_operator_decomposition(
        H=H,
        L=L,
        omegaW=1.0,
        gamma=0.5,
        g=0.1,
        nbar=2.0,
        n_bins=3,
    )

    expected = np.array(
        [
            [
                0.0278171 - 0.01724289j,
                0.02117584 + 0.07920711j,
                -0.01491001 - 0.01669912j,
                -0.04753736 + 0.00105079j,
                0.04529133 - 0.00638164j,
                0.06296986 + 0.00618317j,
                0.00083833 + 0.00518695j,
                -0.01506334 - 0.00536194j,
            ],
            [
                -0.00817447 + 0.03956382j,
                -0.08894369 - 0.04829132j,
                0.02762221 + 0.00082309j,
                0.0394273 - 0.04348076j,
                -0.0327441 + 0.04599552j,
                -0.05901343 + 0.05116358j,
                -0.00535887 - 0.0036537j,
                0.01759551 - 0.0089418j,
            ],
            [
                -0.0051844 + 0.00683826j,
                -0.01263824 - 0.01739056j,
                0.00522078 + 0.00268316j,
                0.01153463 - 0.00473199j,
                -0.01048316 + 0.00582515j,
                -0.01599229 + 0.00441437j,
                -0.00069342 - 0.00119046j,
                0.0041911 - 0.00010576j,
            ],
            [
                -0.01004382 - 0.02016145j,
                0.05562901 - 0.00946442j,
                -0.01239378 + 0.00915357j,
                -0.00230902 + 0.03264374j,
                -0.00148727 - 0.03144424j,
                0.00824973 - 0.04275849j,
                0.00360797 - 0.00024398j,
                -0.00463432 + 0.00998106j,
            ],
            [
                0.00548798 + 0.02319793j,
                -0.05960665 - 0.00366205j,
                0.01497886 - 0.00644378j,
                0.0102514 - 0.03308174j,
                -0.00605016 + 0.03276116j,
                -0.01881232 + 0.04207214j,
                -0.00377676 - 0.00061864j,
                0.00718248 - 0.00916767j,
            ],
            [
                0.02127375 - 0.01809433j,
                0.02796244 + 0.06413401j,
                -0.01470895 - 0.01219001j,
                -0.03997683 + 0.00694519j,
                0.03740228 - 0.01115697j,
                0.05392032 - 0.00280822j,
                0.00136843 + 0.00426976j,
                -0.01339342 - 0.00260446j,
            ],
            [
                0.01522688 + 0.00883685j,
                -0.03223185 + 0.03010503j,
                0.00415068 - 0.01130467j,
                -0.01253475 - 0.0222962j,
                0.01449622 + 0.01988046j,
                0.01300873 + 0.03145236j,
                -0.00224958 + 0.00171118j,
                -0.00126901 - 0.00850698j,
            ],
            [
                -0.01375742 + 0.02200385j,
                -0.04278762 - 0.04894533j,
                0.01645293 + 0.00666335j,
                0.03345543 - 0.01738474j,
                -0.02999167 + 0.02039161j,
                -0.0469675 + 0.01763878j,
                -0.0024118 - 0.00339715j,
                0.01259425 - 0.0014568j,
            ],
        ]
    )
    assert np.isclose(c_ops[1], expected)

    for i, J in enumerate(c_ops):
        assert not np.isnan(J).any(), f"Operator {i} contains NaNs."
        assert not np.isinf(J).any(), f"Operator {i} contains infinities."
        assert np.linalg.norm(J) < 1e6, f"Operator {i} has unreasonably large norm."
