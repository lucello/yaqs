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
