# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""
This module provides funcionality to build jump operators.
"""

import numpy as np
from mqt.yaqs.core.libraries.gate_library import Destroy, GateLibrary, Id, X, Z


def jump_operator_decomposition(
    H: np.ndarray,
    L: np.ndarray,
    omegaW: float,
    gamma: float,
    g: float,
    nbar: float,
    tol: float = 1e-11,
    n_bins: int = 3,
) -> list[np.ndarray]:
    """
    Construct spectrally decomposed jump operators in the eigenbasis of H,
    binned by frequency windows defined by n_bins.

    Args:
        H (np.ndarray): Hamiltonian matrix.
        L (np.ndarray): System operator to decompose.
        omegaW (float): Bath mode freqency.
        gamma (float): Bandwidth of bath mode.
        g (float): System-bath coupling strength.
        nbar (float): Bose bath occupation.
        tol (float): Threshold for keeping matrix elements of L.
        n_bins (int): Number of positive/negative frequency bins.

    Returns:
        list[np.ndarray]: List of spectrally decomposed jump operators.
    """

    # ----------------------------------------------------------------------
    # 1) Eigenbasis of H
    # ----------------------------------------------------------------------
    evals, evecs = np.linalg.eigh(H)  # columns are eigenvectors
    U = evecs  # unitary transformation
    Udag = U.conj().T

    # operator in the eigenbasis
    L_eig = Udag @ L @ U

    # energy differences ω_mn
    omega_mn = evals[:, None] - evals[None, :]

    # mask of significant matrix elements
    Lmask = np.abs(L_eig) > tol

    # ----------------------------------------------------------------------
    # 2) Bin boundaries (same formula as original code)
    # ----------------------------------------------------------------------
    thresholds = np.sqrt((n_bins + 1) / np.arange(1, n_bins + 1) - 1)
    bounds = np.zeros(2 * n_bins)
    bounds[:n_bins] = -thresholds
    bounds[n_bins:] = thresholds[::-1]

    num_bins = len(bounds)
    num_outputs = 2 * (num_bins - 1)

    # allocate output list
    jump_ops = [0 * L for _ in range(num_outputs)]

    # ----------------------------------------------------------------------
    # 3) Loop over signs (emission/absorption) and bins
    # ----------------------------------------------------------------------
    for sign in (-1, 1):  # -1: absorption, +1: emission
        for b in range(1, num_bins):

            # frequency window for this bin
            delta = (omega_mn + sign * omegaW) / gamma
            bin_mask = (delta >= bounds[b - 1]) & (delta < bounds[b]) & Lmask

            if not np.any(bin_mask):
                continue

            # --------------------------------------------------------------
            # 4) Bin‑dependent rate
            # --------------------------------------------------------------
            if sign > 0:  # emission
                center = np.sum(bounds[b - 1 : b]) / 2
                rate = (1 + nbar) * g**2 / gamma / (1 + center**2)
            else:  # absorption
                rate = nbar * g**2 / gamma / (1 + bounds[b] ** 2)

            # --------------------------------------------------------------
            # 5) Construct jump operator in eigenbasis
            # --------------------------------------------------------------
            J_eig = np.zeros_like(L_eig, dtype=complex)
            J_eig[bin_mask] = np.sqrt(rate) * L_eig[bin_mask]

            # transform back to lab basis once
            J_lab = U @ J_eig @ Udag

            # output index: first half = sign<0, second half = sign>0
            out_idx = (b - 1) + (num_bins - 1) * (sign == 1)
            jump_ops[out_idx] = J_lab

    return jump_ops


def bose_hubbard_dense(
    length: int, local_dim: int, omega: float, hopping_j: float, hubbard_u: float
) -> np.ndarray:
    """Construct the exact dense Bose-Hubbard Hamiltonian for comparison.

    Returns:
        Dense Hamiltonian matrix.
    """
    # Local operators
    a = Destroy(local_dim).matrix
    adag = Destroy(local_dim).dag().matrix
    n = adag @ a
    id_op = np.eye(local_dim, dtype=complex)

    dim = local_dim**length
    H = np.zeros((dim, dim), dtype=complex)

    # Build H term-by-term using Kronecker products
    def embed(op_list: list[np.ndarray]) -> np.ndarray:
        out = np.array([[1.0]], dtype=complex)
        for op in op_list:
            out = np.kron(out, op)
        return out

    # Onsite terms
    for i in range(length):
        op_list = [id_op] * length
        op_list[i] = omega * n + 0.5 * hubbard_u * (n @ (n - id_op))
        H += embed(op_list)

    # Hopping terms
    for i in range(length - 1):
        # adag_i * a_{i+1}
        op_list1 = [id_op] * length
        op_list1[i] = adag
        op_list1[i + 1] = a
        H += -hopping_j * embed(op_list1)

        # a_i * adag_{i+1}
        op_list2 = [id_op] * length
        op_list2[i] = a
        op_list2[i + 1] = adag
        H += -hopping_j * embed(op_list2)

    return H


import numpy as np


def jumps_bose_hubbard_cooling(
    length: int,
    local_dim: int,
    omega: float,
    hopping_j: float,
    hubbard_u: float,
    omega_w: float,
    width: float,
    coupling: float,
    bose_factor: float,
):
    """
    Construct spectrally decomposed jump operators in the eigenbasis of H,
    binned by frequency windows defined by `n_bins`.

    Args:
        H (np.ndarray): Hamiltonian matrix.
        L (np.ndarray): System operator to decompose.
        omega_w (float): Bath mode frequency.
        gamma (float): Bandwidth of the bath mode.
        g (float): System–bath coupling strength.
        nbar (float): Bose–Einstein occupation of the bath.
        tol (float): Threshold for discarding small matrix elements of L.
        n_bins (int): Number of positive/negative frequency bins.

    Returns:
        list[np.ndarray]: List of spectrally decomposed jump operators.
    """

    # 1) Dense Bose–Hubbard Hamiltonian in the requested notation
    H = bose_hubbard_dense(
        length=length,
        local_dim=local_dim,
        omega=omega,
        hopping_j=hopping_j,
        hubbard_u=hubbard_u,
    )

    # 2) Diagonalize with NumPy (Hermitian → eigh). No QuTiP dependence.
    evals, _ = np.linalg.eigh(H)  # evecs columns are eigenvectors

    # 3) Local operators and embedding helper (same style as your H builder)
    a = Destroy(local_dim).matrix
    adag = Destroy(local_dim).dag().matrix
    n_loc = adag @ a
    id_op = np.eye(local_dim, dtype=complex)

    def embed(op_list: list[np.ndarray]) -> np.ndarray:
        out = np.array([[1.0]], dtype=complex)
        for op in op_list:
            out = np.kron(out, op)
        return out

    # Build n_j operators using the same Kronecker style
    n_ops: list[np.ndarray] = []
    for j in range(length):
        op_list = [id_op] * length
        op_list[j] = n_loc
        n_ops.append(embed(op_list))

    # 4) Build jump operators by decomposing a chosen set of local ops
    c_ops: list[np.ndarray] = []

    for n_j in n_ops:
        c_ops.extend(
            jump_operator_decomposition(
                H,
                n_j,
                omega_w,
                width,
                coupling,
                bose_factor,
            )
        )

    return c_ops
