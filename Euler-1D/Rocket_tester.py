### DE RESCRIS INFO DESPRE PROGRAM ###
# Acest cod simulează ecuațiile Euler 1D pentru o problemă Riemann.
# Codul inițializează o problemă Riemann cu o discontinuitate în densitate și presiune, aplică condiții la margine reflective și afișează
#   profilurile finale de densitate, presiune și viteză la sfârșitul simulării.
# Functionalitati principale: 
# - Reconstrucție WENO5 caracteristică pentru variabilele conservate
# - Estimări ale vitezei maxime de undă folosind metodele Davidson și Toro (Toro superior, putin mai costisitor)
# - Flux HLLC folosit pentru calculul fluxurilor numerice
# - Pas de timp dt calculat dupa fiecare iteratie
# - Metoda de integrare în timp SSPRK45 pentru avansarea soluției în timp (5 pasi, aproape dublu costul RK3)

from __future__ import annotations
import os
from Riemann_test_cases import test_case
from input_VegaE import *

# Hint BLAS/OpenMP backends to use all CPU threads for vectorized kernels.
os.environ.setdefault("OMP_NUM_THREADS", "20")
os.environ.setdefault("MKL_NUM_THREADS", "20")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "20")

import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

ar = 'auto'
height = 16
width = 9
dpi = 100
title_size = 28
label_size = 28
tick_size = 20
legend_size = 28
line_color = 'orangered'


def simulate():
    real_cell_slice = slice(3, -4)
    updated_cell_slice = slice(3, -3)
    fictitious_cell_idx = -4
    nozzle_state_idx = -3
    area_floor = 1e-12
    rho_floor = max(1e-9, 1e-8 * rho0)
    p_floor = max(1.0, 1e-8 * p0)
    a2_floor = 1e-10

    def regularize_denom(x, eps=1e-12):
        return np.where(np.abs(x) < eps, np.where(x >= 0.0, eps, -eps), x)

    def enforce_positive_state(U, A, k):
        """Project conservative state to rho>0 and p>0 while preserving momentum.
        Vectorized with reused sub-expressions for efficiency.
        """
        Ause = np.maximum(A, area_floor)
        rho = np.maximum(U[0, :] / Ause, rho_floor)
        rho_Ause = rho * Ause
        u = U[1, :] / rho_Ause

        kminus1 = np.maximum(k - 1.0, 1e-8)
        E = U[2, :] / rho_Ause
        u2 = u * u
        p = rho * kminus1 * (E - 0.5 * u2)
        p = np.maximum(p, p_floor)
        E_final = p / (rho * kminus1) + 0.5 * u2

        U[0, :] = rho * Ause
        U[1, :] = u * rho_Ause  # Directly compute from u and rho*Ause
        U[2, :] = rho * E_final * Ause
        return U
    
    def primitives(U, A, k):
        """Convert conserved variables U to primitive variables (rho, u, p, E, a).
        U = [rho*A, rho*u*A, rho*E*A], where E is total energy per volume. A is cross-sectional area.
        Vectorized with minimal temporary allocations for multi-thread efficiency.
        """

        if U.shape[1] != A.shape[0]:
            Ause = A[2:-2]  # trim A to match U's spatial dimension if needed
            Ause = 0.5*(Ause[:-1] + Ause[1:])  # average adjacent A values to get A at cell centers for better accuracy in primitives
        else:
            Ause = A
        Ause = np.maximum(Ause, area_floor)

        rho = np.maximum(U[0, :] / Ause, rho_floor)
        rho_Ause = rho * Ause
        u = U[1, :] / rho_Ause
        E = U[2, :] / rho_Ause    # total specific energy (per unit mass)
        
        kminus1 = np.maximum(k - 1.0, 1e-8)
        u2 = u * u
        p = rho * kminus1 * (E - 0.5 * u2)
        p = np.maximum(p, p_floor)
        # Recompute E once with final p (avoid intermediate E recomputation)
        E_final = p / (rho * kminus1) + 0.5 * u2
        a2 = np.maximum(k * p / rho, a2_floor)
        a = np.sqrt(a2)
        return rho, u, p, E_final, a
    
    def initial_Riemann(U, A, left_initial, right_initial, k):
        shape = U.shape
        Aext = np.zeros([3, shape[1]], dtype=np.float64)
        Aext = np.tile(A, (3, 1))  # extend A to match U's spatial dimension if needed

        rhoL = left_initial[0]
        pL = left_initial[1]
        uL = left_initial[2]
        kL = k[0]
        left_state = np.array([rhoL, rhoL*uL, (pL/(kL-1) + 0.5*rhoL*uL*uL)])  # (rho*A, rho*u*A, rho*E*A) left state

        rhoR = right_initial[0]
        pR = right_initial[1]
        uR = right_initial[2]
        kR = k[-1]
        right_state = np.array([rhoR, rhoR*uR, (pR/(kR-1) + 0.5*rhoR*uR*uR)])  # (rho*A, rho*u*A, rho*E*A) right state
        
        UL = np.repeat(left_state, shape[1]//2, axis=0).reshape(3, shape[1]//2)
        UR = np.repeat(right_state, shape[1]//2, axis=0).reshape(3, shape[1]//2)

        if shape[1] % 2 != 0:
            U[:, :shape[1]//2, 0] = UL*Aext[:, :shape[1]//2]
            U[:, shape[1]//2, 0] = Aext[:, shape[1]//2]*(UL[:, -1] + UR[:, 0]) / 2  # set middle point to average of left and right states for better WENO reconstruction
            U[:, shape[1]//2+1:, 0] = UR*Aext[:, shape[1]//2+1:]
        else: 
            U[:, :shape[1]//2, 0] = UL*Aext[:, :shape[1]//2]
            U[:, shape[1]//2:, 0] = UR*Aext[:, shape[1]//2:]
        return U

    def gas_constant_from_k(k_local):
        if np.isscalar(k_local):
            if abs(k_local - kgas) <= abs(k_local - kigniter):
                return Rgas
            return Rgas_igniter
        return np.where(np.abs(k_local - kgas) <= np.abs(k_local - kigniter), Rgas, Rgas_igniter)

    def apply_atmospheric_outlet(U, A, k):
        i_in = fictitious_cell_idx # LLEN2F
        rho_in = max(U[0, i_in] / max(A[i_in], area_floor), rho_floor)
        u_in = U[1, i_in] / (rho_in * A[i_in])
        E_in = U[2, i_in] / (rho_in * A[i_in])
        k_in = k[i_in]
        p_in = rho_in * max(k_in - 1.0, 1e-8) * (E_in - 0.5 * u_in * u_in)
        p_in = max(p_in, p_floor)
        a_in = np.sqrt(max(k_in * p_in / rho_in, a2_floor))
        if u_in >= a_in:
            U[:, nozzle_state_idx:] = U[:, i_in:i_in+1]
            k[nozzle_state_idx:] = k[i_in:i_in+1]
            return enforce_positive_state(U, A, k), k

        if u_in >= 0.0:
            rho_g = rho_in
            u_g = u_in
            p_g = p0
        else:
            rho_g = rho0
            u_g = 0.0
            p_g = p0

        E_g = p_g / (k_in - 1.0) + 0.5 * rho_g * u_g * u_g
        U[0, nozzle_state_idx:] = rho_g * A[nozzle_state_idx:]
        U[1, nozzle_state_idx:] = rho_g * u_g * A[nozzle_state_idx:]
        U[2, nozzle_state_idx:] = E_g * A[nozzle_state_idx:]
        k[nozzle_state_idx:] = k_in
        return enforce_positive_state(U, A, k), k

    def apply_nozzle_outlet(U, A, k):
        rho_f = max(U[0, fictitious_cell_idx] / max(A[fictitious_cell_idx], area_floor), rho_floor)
        u_f = U[1, fictitious_cell_idx] / (rho_f * A[fictitious_cell_idx])
        E_f = U[2, fictitious_cell_idx] / (rho_f * A[fictitious_cell_idx])
        k_f = k[fictitious_cell_idx]
        p_f = rho_f * max(k_f - 1.0, 1e-8) * (E_f - 0.5 * u_f * u_f)

        if not np.isfinite(rho_f) or not np.isfinite(u_f) or not np.isfinite(p_f):
            return apply_atmospheric_outlet(U, A, k)
        if rho_f <= 1e-12 or p_f <= max(1.0, p0) or u_f <= 0.0:
            return apply_atmospheric_outlet(U, A, k)

        R_f = gas_constant_from_k(k_f)
        T_f = p_f / (rho_f * R_f)
        a_f = np.sqrt(max(k_f * p_f / rho_f, a2_floor))
        M_f = max(u_f / a_f, 0.0)

        T0_f = T_f * (1.0 + 0.5 * (k_f - 1.0) * M_f * M_f)
        p0_f = p_f * (1.0 + 0.5 * (k_f - 1.0) * M_f * M_f) ** (k_f / (k_f - 1.0))

        Athroat = A2nozzle / epsnozzle
        area_ratio = A[fictitious_cell_idx] / Athroat
        if area_ratio <= 1.0:
            return apply_atmospheric_outlet(U, A, k)

        M_s = solve_area_mach(area_ratio, k_f, supersonic=True)
        T_s = T0_f / (1.0 + 0.5 * (k_f - 1.0) * M_s * M_s)
        p_s = p0_f / (1.0 + 0.5 * (k_f - 1.0) * M_s * M_s) ** (k_f / (k_f - 1.0))
        rho_s = p_s / (R_f * T_s)
        u_s = M_s * np.sqrt(k_f * R_f * T_s)

        if not np.isfinite(rho_s) or not np.isfinite(u_s) or not np.isfinite(p_s):
            return apply_atmospheric_outlet(U, A, k)
        if rho_s <= 1e-12 or p_s <= 1.0:
            return apply_atmospheric_outlet(U, A, k)

        E_s = p_s / (k_f - 1.0) + 0.5 * rho_s * u_s * u_s
        U[0, nozzle_state_idx:] = rho_s * A[nozzle_state_idx:]
        U[1, nozzle_state_idx:] = rho_s * u_s * A[nozzle_state_idx:]
        U[2, nozzle_state_idx:] = E_s * A[nozzle_state_idx:]
        k[nozzle_state_idx:] = k_f
        return U, k


    def Riemann_BC(U, A, k):

        # Left boundary: rigid reflective wall.
        left_dst = np.array([2, 1, 0])
        left_src = np.array([3, 4, 5])
        U[:, left_dst] = U[:, left_src]
        U[1, left_dst] = -U[1, left_src]

        if boundary_case != 'wall-atmosphere':
            raise ValueError("boundary_case must be 'wall-atmosphere' for accurate depiction of rocket flow.")
        return apply_nozzle_outlet(U, A, k)
    
    def weno5_reconstruct(U, A, k):
        """
        Characteristic WENO5 reconstruction for 1D Euler variables.
        Optimized for multi-threaded BLAS with vectorized ENO3 operations.

        U shape: (nvar, nx), with at least 3 ghost cells on each side.
        Returns arrays with shape (nvar, nx-6).
        """
        nvar, nx = U.shape
        if nx < 7:
            raise ValueError("Need at least 7 points (including ghost cells) for WENO5.")
        if nvar != 3:
            raise ValueError("Characteristic WENO5 here is implemented for 1D Euler with 3 conserved variables.")

        eps_weno = 1e-6
        m = nx - 5  # number of reconstructed interfaces

        # Build all stencils at once to avoid Python-loop overhead.
        u_im3 = U[:, 0:m]
        u_im2 = U[:, 1:m+1]
        u_im1 = U[:, 2:m+2]
        u_i = U[:, 3:m+3]
        u_ip1 = U[:, 4:m+4]
        u_ip2 = U[:, 5:m+5]

        A_im1 = A[2:m+2]  # A at i-1 for primitives calculation
        A_i = A[3:m+3]    # A at i for primitives calculation
        A_im1_safe = np.maximum(A_im1, area_floor)
        A_i_safe = np.maximum(A_i, area_floor)

        # Compute left and right primitives with reused sub-expressions
        rhoL = np.maximum(u_im1[0] / A_im1_safe, rho_floor)
        rhoL_A = rhoL * A_im1_safe
        velL = u_im1[1] / rhoL_A
        EL = u_im1[2] / rhoL_A
        kL = k[2:m+2]
        kLm1 = np.maximum(kL - 1.0, 1e-8)
        velL2 = velL * velL
        pL = rhoL * kLm1 * (EL - 0.5 * velL2)
        pL = np.maximum(pL, p_floor)
        HL = EL + pL / rhoL

        rhoR = np.maximum(u_i[0] / A_i_safe, rho_floor)
        rhoR_A = rhoR * A_i_safe
        velR = u_i[1] / rhoR_A
        ER = u_i[2] / rhoR_A
        kR = k[3:m+3]
        kRm1 = np.maximum(kR - 1.0, 1e-8)
        velR2 = velR * velR
        pR = rhoR * kRm1 * (ER - 0.5 * velR2)
        pR = np.maximum(pR, p_floor)
        HR = ER + pR / rhoR

        sL = np.sqrt(np.maximum(rhoL, rho_floor))
        sR = np.sqrt(np.maximum(rhoR, rho_floor))
        denom = regularize_denom(sL + sR)
        inv_denom = 1.0 / denom

        rhohat = sL * sR
        uhat = (sL * velL + sR * velR) * inv_denom
        Hhat = (sL * HL + sR * HR) * inv_denom
        khat = (sL * kL + sR * kR) * inv_denom
        uhat2 = uhat * uhat
        ahat2 = (khat - 1.0) * (Hhat - 0.5 * uhat2)
        ahat2 = np.maximum(ahat2, a2_floor)
        ahat = np.sqrt(ahat2)
        
        # Precompute common terms once
        rho_2a = rhohat / (2.0 * ahat)
        rho_u_a_2a = rhohat * (uhat + ahat) / (2.0 * ahat)
        rho_u_m_a_2a = rhohat * (uhat - ahat) / (2.0 * ahat)
        H_u_a = Hhat + uhat * ahat
        H_u_m_a = Hhat - uhat * ahat

        # Build P matrix (batch inversion via LAPACK backend)
        Pmat = np.empty((m, 3, 3), dtype=U.dtype)
        Pmat[:, 0, :] = np.column_stack((np.ones(m), rho_2a, rho_2a))
        Pmat[:, 1, :] = np.column_stack((uhat, rho_u_a_2a, rho_u_m_a_2a))
        Pmat[:, 2, :] = np.column_stack((0.5 * uhat2, rhohat * H_u_a / (2.0 * ahat), rhohat * H_u_m_a / (2.0 * ahat)))

        # Batch invert (multi-threaded LAPACK)
        L = np.linalg.inv(Pmat)
        eigenvals = np.vstack((uhat, uhat + ahat, uhat - ahat))

        # Characteristic projections via batch matmul (multi-threaded BLAS dgemm)
        wm_im3 = np.einsum('mab,bm->am', L, u_im3)
        wm_im2 = np.einsum('mab,bm->am', L, u_im2)
        wm_im1 = np.einsum('mab,bm->am', L, u_im1)
        wm_i = np.einsum('mab,bm->am', L, u_i)
        wm_ip1 = np.einsum('mab,bm->am', L, u_ip1)
        wm_ip2 = np.einsum('mab,bm->am', L, u_ip2)

        # Vectorized ENO3 stencil operations (no nested loops)
        # Left reconstruction
        p0_L = (1.0 / 3.0) * wm_im3 - (7.0 / 6.0) * wm_im2 + (11.0 / 6.0) * wm_im1
        p1_L = -(1.0 / 6.0) * wm_im2 + (5.0 / 6.0) * wm_im1 + (1.0 / 3.0) * wm_i
        p2_L = (1.0 / 3.0) * wm_im1 + (5.0 / 6.0) * wm_i - (1.0 / 6.0) * wm_ip1
        
        dw_L1 = wm_im3 - 2*wm_im2 + wm_im1
        dw_L2 = wm_im2 - 2*wm_im1 + wm_i
        dw_L3 = wm_im1 - 2*wm_i + wm_ip1
        
        beta1_L = (13.0/12.0)*dw_L1**2 + (1.0/4.0)*(wm_im3 - 4*wm_im2 + 3*wm_im1)**2
        beta2_L = (13.0/12.0)*dw_L2**2 + (1.0/4.0)*(wm_im2 - wm_i)**2
        beta3_L = (13.0/12.0)*dw_L3**2 + (1.0/4.0)*(3*wm_im1 - 4*wm_i + wm_ip1)**2
        
        alpha1_L = (1.0/10.0) / (eps_weno + beta1_L)**2
        alpha2_L = (3.0/5.0) / (eps_weno + beta2_L)**2
        alpha3_L = (3.0/10.0) / (eps_weno + beta3_L)**2
        sum_alpha_L = alpha1_L + alpha2_L + alpha3_L
        sum_alpha_L_inv = 1.0 / sum_alpha_L
        w1_L = alpha1_L * sum_alpha_L_inv
        w2_L = alpha2_L * sum_alpha_L_inv
        w3_L = 1.0 - w1_L - w2_L
        
        w_uL = w1_L*p0_L + w2_L*p1_L + w3_L*p2_L

        # Right reconstruction
        p0_R = -(1.0 / 6.0) * wm_im2 + (5.0 / 6.0) * wm_im1 + (1.0 / 3.0) * wm_i
        p1_R = (1.0 / 3.0) * wm_im1 + (5.0 / 6.0) * wm_i - (1.0 / 6.0) * wm_ip1
        p2_R = (11.0 / 6.0) * wm_i - (7.0 / 6.0) * wm_ip1 + (1.0 / 3.0) * wm_ip2
        
        dw_R1 = wm_im2 - 2*wm_im1 + wm_i
        dw_R2 = wm_im1 - 2*wm_i + wm_ip1
        dw_R3 = wm_i - 2*wm_ip1 + wm_ip2
        
        beta1_R = (13.0/12.0)*dw_R1**2 + (1.0/4.0)*(wm_im2 - 4*wm_im1 + 3*wm_i)**2
        beta2_R = (13.0/12.0)*dw_R2**2 + (1.0/4.0)*(wm_im1 - wm_ip1)**2
        beta3_R = (13.0/12.0)*dw_R3**2 + (1.0/4.0)*(3*wm_i - 4*wm_ip1 + wm_ip2)**2
        
        alpha1_R = (3.0/10.0) / (eps_weno + beta1_R)**2
        alpha2_R = (3.0/5.0) / (eps_weno + beta2_R)**2
        alpha3_R = (1.0/10.0) / (eps_weno + beta3_R)**2
        sum_alpha_R = alpha1_R + alpha2_R + alpha3_R
        sum_alpha_R_inv = 1.0 / sum_alpha_R
        w1_R = alpha1_R * sum_alpha_R_inv
        w2_R = alpha2_R * sum_alpha_R_inv
        w3_R = 1.0 - w1_R - w2_R
        
        w_uR = w1_R*p0_R + w2_R*p1_R + w3_R*p2_R

        # Transform back via batch matmul
        UL = np.einsum('mab,bm->am', Pmat, w_uL)
        UR = np.einsum('mab,bm->am', Pmat, w_uR)
        return UL, UR, eigenvals, khat
    
    # Wave speed
    def max_wave_speed_Toro(U, A, case, k):
        """Compute max wave speeds using Toro's method. Optimized for batch operations."""
        UL, UR, _, khat = weno5_reconstruct(U, A, k)

        rhoL, uL, pL, _, aL = primitives(UL, A, khat)
        rhoR, uR, pR, _, aR = primitives(UR, A, khat)

        pL = np.maximum(pL, p_floor)
        pR = np.maximum(pR, p_floor)
        khat_safe = np.maximum(khat, 1.0 + 1e-6)

        khat_m1 = khat_safe - 1.0
        khat_p1 = khat_safe + 1.0
        inv_2khat = 1.0 / (2.0 * khat_safe)
        
        gamma_exp = khat_m1 * inv_2khat
        power_exp = 2.0 * khat_safe / khat_m1
        
        # Compute p* estimate (avoid repeated exponentiation)
        pL_gamma = np.power(pL, gamma_exp)
        pR_gamma = np.power(pR, gamma_exp)
        base_num = aL + aR - 0.5 * khat_m1 * (uR - uL)
        base_den = aL / pL_gamma + aR / pR_gamma
        base = np.maximum(base_num / regularize_denom(base_den), 1e-16)
        pstarr = np.maximum(np.power(base, power_exp), p_floor)

        # Precompute constants for q factor
        coeff = khat_p1 * inv_2khat
        
        # Compute q factors (shock-capturing)
        pstarr_pL = pstarr / pL
        pstarr_pR = pstarr / pR
        
        qL_arg = np.maximum(1.0 + coeff * (pstarr_pL - 1.0), 1.0)
        qR_arg = np.maximum(1.0 + coeff * (pstarr_pR - 1.0), 1.0)
        
        qL = np.where(pstarr <= pL, 1.0, np.sqrt(qL_arg))
        qR = np.where(pstarr <= pR, 1.0, np.sqrt(qR_arg))

        SL = uL - aL * qL
        SR = uR + aR * qR
        
        if case == 'dt':
            return np.max(np.abs([SL, SR]))
        elif case == 'flux':
            return SL, SR

    def Euler_flux(U, A, k):
        """Compute Euler flux with optimized primitive computation."""
        if U.shape[1] != A.shape[0]:
            Ause = A[2:-2]  
            Ause = 0.5*(Ause[:-1] + Ause[1:])  
        else:
            Ause = A
        Ause = np.maximum(Ause, area_floor)  # Ensure no zero areas
        
        rho, u, p, E, _ = primitives(U, A, k)
        u2_rho = u * rho
        u_rho_E_plus_p = u * (rho * E + p)
        return np.vstack((u2_rho * Ause, (u2_rho * u + p) * Ause, u_rho_E_plus_p * Ause))

    def HLLE_flux(U, A, k):
        """
        Low-order HLLE flux: robust and positivity-preserving but non-dissipative.
        Used as fallback for troubled interfaces.
        """
        UL, UR, _, khat = weno5_reconstruct(U, A, k)
        rhoL, uL, pL, EL, aL = primitives(UL, A, khat)
        rhoR, uR, pR, ER, aR = primitives(UR, A, khat)

        fL = Euler_flux(UL, A, khat)
        fR = Euler_flux(UR, A, khat)

        Ause = A[2:-2]
        Ause = np.maximum(0.5*(Ause[:-1] + Ause[1:]), area_floor)

        SL, SR = max_wave_speed_Toro(U, A, 'flux', k)

        # HLLE: simple two-wave HLL with no intermediate state
        # f = (SR*fL - SL*fR + SL*SR*(UR - UL)) / (SR - SL)
        denom_hlle = regularize_denom(SR - SL)
        f_hlle = (SR * fL - SL * fR + SL * SR * (UR - UL)) / denom_hlle

        fm = f_hlle[:, :-1]
        fp = f_hlle[:, 1:]
        return fm, fp

    def detect_troubled_interfaces(U, A, k):
        """
        Detect troubled interfaces where positivity is at risk.
        Optimized for vectorized computation across all interfaces.
        Returns a (nx-6,) boolean array of troubled interface locations.
        """
        UL, UR, _, khat = weno5_reconstruct(U, A, k)
        
        # Floor primitives to check where they'd be negative before flooring
        Ause_l = np.maximum(A[2:-3], area_floor)
        Ause_r = np.maximum(A[3:-2], area_floor)
        
        rhoL_unsafe = UL[0, :] / Ause_l
        rhoR_unsafe = UR[0, :] / Ause_r
        
        # Check for density approaching floor
        troubled = (rhoL_unsafe < 2.0 * rho_floor) | (rhoR_unsafe < 2.0 * rho_floor)
        
        # Check pressure safety (vectorized)
        rhoL = np.maximum(rhoL_unsafe, rho_floor)
        rhoR = np.maximum(rhoR_unsafe, rho_floor)
        rhoL_Ause_l = rhoL * Ause_l
        rhoR_Ause_r = rhoR * Ause_r
        
        velL2 = (UL[1, :] / rhoL_Ause_l) ** 2
        velR2 = (UR[1, :] / rhoR_Ause_r) ** 2
        
        EL = UL[2, :] / rhoL_Ause_l
        ER = UR[2, :] / rhoR_Ause_r
        
        kL = k[2:-3]
        kR = k[3:-2]
        kLm1 = np.maximum(kL - 1.0, 1e-8)
        kRm1 = np.maximum(kR - 1.0, 1e-8)
        
        pL_unsafe = rhoL * kLm1 * (EL - 0.5 * velL2)
        pR_unsafe = rhoR * kRm1 * (ER - 0.5 * velR2)
        
        troubled |= (pL_unsafe < 2.0 * p_floor) | (pR_unsafe < 2.0 * p_floor)
        
        # Large jumps in pressure (shock detection) - vectorized
        pL_safe = np.maximum(pL_unsafe, p_floor)
        pR_safe = np.maximum(pR_unsafe, p_floor)
        p_sum = pL_safe + pR_safe
        pressure_jump = np.abs(pL_safe - pR_safe) / np.maximum(0.5 * p_sum, p_floor)
        troubled |= pressure_jump > 0.5  # 50% jump threshold
        
        return troubled

    def blend_flux(fm_hllc, fp_hllc, fm_hlle, fp_hlle, troubled):
        """
        Blend HLLC and HLLE fluxes based on troubled cell indicators.
        Vectorized implementation without Python loops.
        Returns blended flux tuple (fm_blend, fp_blend).
        """
        # Initialize theta array (fully HLLC)
        n_interfaces = fm_hllc.shape[1] + 1
        theta = np.ones(n_interfaces, dtype=np.float64)
        theta[troubled] = 0.0  # Full HLLE at troubled interface
        
        # Vectorized neighbor blending: neighbors of troubled cells get 0.5
        # Left neighbors of troubled cells
        troubled_indices = np.where(troubled)[0]
        left_neighbors = troubled_indices - 1
        left_neighbors = left_neighbors[left_neighbors >= 0]
        theta[left_neighbors] = np.minimum(theta[left_neighbors], 0.5)
        
        # Right neighbors of troubled cells
        right_neighbors = troubled_indices + 1
        right_neighbors = right_neighbors[right_neighbors < n_interfaces]
        theta[right_neighbors] = np.minimum(theta[right_neighbors], 0.5)
        
        # Blend: f_blend = theta * f_hllc + (1 - theta) * f_hlle
        theta_m = theta[:-1]
        theta_p = theta[1:]
        one_minus_theta_m = 1.0 - theta_m
        one_minus_theta_p = 1.0 - theta_p
        
        fm_blend = theta_m * fm_hllc + one_minus_theta_m * fm_hlle
        fp_blend = theta_p * fp_hllc + one_minus_theta_p * fp_hlle
        
        return fm_blend, fp_blend

    def HLLC_flux(U, A, k):
        UL, UR, eigenvals, khat = weno5_reconstruct(U, A, k)
        rhoL, uL, pL, EL, aL = primitives(UL, A, khat)  # primitives at left state
        rhoR, uR, pR, ER, aR = primitives(UR, A, khat)  # primitives at right state

        fL = Euler_flux(UL, A, khat)
        fR = Euler_flux(UR, A, khat)

        Ause = A[2:-2]
        Ause = np.maximum(0.5*(Ause[:-1] + Ause[1:]), area_floor)

        SL, SR = max_wave_speed_Toro(U, A, 'flux', k)  # Toro's max wave speeds at x_{i-1/2} and x_{i+1/2}

        denom_sstar = regularize_denom(rhoL*(SL-uL) - rhoR*(SR-uR))
        Sstar = (pR - pL + rhoL*uL*(SL-uL) - rhoR*uR*(SR-uR)) / denom_sstar

        denom_l = regularize_denom(SL - Sstar)
        denom_r = regularize_denom(SR - Sstar)
        denom_e_l = regularize_denom(rhoL*(SL-uL))
        denom_e_r = regularize_denom(rhoR*(SR-uR))

        UstarL = Ause * rhoL * (SL - uL) / denom_l * np.array([np.ones(SL.shape), Sstar, EL + (Sstar - uL)*(Sstar + pL/denom_e_l)])
        UstarR = Ause * rhoR * (SR - uR) / denom_r * np.array([np.ones(SR.shape), Sstar, ER + (Sstar - uR)*(Sstar + pR/denom_e_r)])
        fstarL = fL + SL * (UstarL - UL)
        fstarR = fR + SR * (UstarR - UR)

        f = np.zeros_like(UL)

        # Vectorized HLLC state selection (replaces scalar Python loop).
        mask_L = SL >= 0.0
        mask_starL = (SL < 0.0) & (Sstar >= 0.0)
        mask_starR = (SR > 0.0) & (Sstar <= 0.0)
        mask_R = SR <= 0.0

        f[:, mask_L] = fL[:, mask_L]
        f[:, mask_starL] = fstarL[:, mask_starL]
        f[:, mask_starR] = fstarR[:, mask_starR]
        f[:, mask_R] = fR[:, mask_R]

        fm = f[:, :-1]  # flux at i-1/2
        fp = f[:, 1:]   # flux at i+1/2

        # Detect troubled interfaces and blend with HLLE where needed
        troubled = detect_troubled_interfaces(U, A, k)
        fm_hlle, fp_hlle = HLLE_flux(U, A, k)
        fm, fp = blend_flux(fm, fp, fm_hlle, fp_hlle, troubled)

        return fm, fp

    def AP_map(A0, dtrb):
        """Map from area A and erosion rate dtrb to new area and perimeter.
        Optimized by reducing sqrt calls and precomputing constants."""
        r0 = np.sqrt(A0) / np.sqrt(np.pi)  # Separate sqrt for potential vectorization
        r1 = r0 + dtrb
        A1 = np.pi * r1 * r1  # Avoid **2 for clarity and potential optimization
        P1 = 2.0 * np.sqrt(A1 * np.pi)
        return A1, P1

    def fill_geometry_ghosts(A, P):
        # linear extrapolation for better accuracylinear extrapolation for better accuracy
        # A[2] = A[3] - (A[4] - A[3])
        # A[1] = A[3] - 2*(A[4] - A[3])
        # A[0] = A[3] - 3*(A[4] - A[3])
        A[0:3] = A[3]
        # print(A[0:10])

        # A[-3] = A[-4] - (A[-5] - A[-4])
        # A[-2] = A[-4] - 2*(A[-5] - A[-4])
        # A[-1] = A[-4] - 3*(A[-5] - A[-4])
        A[-4:] = A[-5]

        # P[2] = P[3] - (P[4] - P[3])
        # P[1] = P[3] - 2*(P[4] - P[3])
        # P[0] = P[3] - 3*(P[4] - P[3])
        P[0:3] = P[3]

        # P[-3] = P[-4] - (P[-5] - P[-4])
        # P[-2] = P[-4] - 2*(P[-5] - P[-4])
        # P[-1] = P[-4] - 3*(P[-5] - P[-4])
        P[-4:] = P[-5]

        return A, P
    
    def erosive_burning(U, A, P, dt, k):
        rho, u, p, _, _ = primitives(U, A, k)

        G = rho[real_cell_slice] * u[real_cell_slice]
        Dh = 4 * A[real_cell_slice] / P[real_cell_slice]
        rb0 = arb * (p[real_cell_slice] ** nrb)
        r = np.zeros_like(rb0)
        indextrue = np.where((G > 1e-9) & (Dh > 1e-9))
        # print(G[0:10])
        if (G > 1e-9).any() and (Dh > 1e-9).any():
            # Fixed-point iterate r = r0 + alpha * G^0.8 * D^-0.2 * exp(-beta * rho_p * r / G)
            re_scale = (G[indextrue] ** 0.8) * (Dh[indextrue] ** -0.2)
            r_iter = rb0[indextrue].copy()
            for _ in range(12):
                expo = -beta_er * rhosolid * r_iter / G[indextrue]
                expo = np.where(expo < -60.0, -60.0, expo)
                re = alpha_er * re_scale * np.exp(expo)
                r_new = rb0[indextrue] + re
                converged = np.abs(r_new - r_iter) <= 1e-6 * np.maximum(1e-4, r_new)
                if np.all(converged):
                    r_iter = r_new
                    break
                r_iter = r_new
            r[indextrue] = np.maximum(0.0, r_iter)
        else:
            r = rb0

        return r

    def init_solid_thermal_state(nx):
        n_inner = nx - 7
        state = {
            'Ts': np.full(n_inner, Tp0, dtype=np.float64),
            'ignited': np.zeros(n_inner, dtype=bool),
            'q_hist': [],
            't_hist': [0.0],
            'time': 0.0,
        }
        return state

    def update_ignition_thermal_state(U, A, P, thermal_state, dt, k):
        rho, u, p, _, _ = primitives(U, A, k)
        rho_i = rho[real_cell_slice]
        p_i = p[real_cell_slice]

        R = np.zeros_like(p_i)
        R[np.where(thermal_state['ignited'])] = Rgas
        R[np.where(~thermal_state['ignited'])] = Rgas_igniter

        mu = np.zeros_like(p_i)
        mu[np.where(thermal_state['ignited'])] = mugas
        mu[np.where(~thermal_state['ignited'])] = mugas_igniter

        K = np.zeros_like(p_i)
        K[np.where(thermal_state['ignited'])] = Kgas
        K[np.where(~thermal_state['ignited'])] = Kgas_igniter

        cp = np.zeros_like(p_i)
        cp[np.where(thermal_state['ignited'])] = cpgas
        cp[np.where(~thermal_state['ignited'])] = cpgas_igniter

        Tg = p_i / (rho_i * R)
        # print(Tg)

        Dh = 4.0 * A[real_cell_slice] / P[real_cell_slice]
        # Re = rho_i * np.abs(u[3:-3]) * Dh / np.maximum(mu_gas_igniter, 1e-12)
        Re = rho_i * np.abs(u[real_cell_slice]) * Dh / mu
        # Pr = np.maximum(cp_gas_igniter * mu_gas_igniter / np.maximum(K_gas_igniter, 1e-12), 1e-6)
        Pr = cp * mu / K
        Nu = np.where(Re > 2300.0, 0.023 * np.power(Re, 0.8) * np.power(Pr, 0.4), 3.66)
        # htc = np.maximum(Nu * K_gas_igniter / Dh, 1.0)
        htc = Nu * K / Dh

        alpha_s = Ksolid / (rhosolid * cpsolid)
        coeff = 2.0 * np.sqrt(alpha_s) / (Ksolid * np.sqrt(np.pi))

        t_old = float(thermal_state['time'])
        t_new = t_old + dt
        q_hist = thermal_state['q_hist']
        t_hist = thermal_state['t_hist']

        history_term = np.zeros_like(Tg)
        if len(q_hist) > 0:
            # Vectorize historical heat flux accumulation (avoid loop overhead)
            q_arr = np.column_stack(q_hist)  # shape (n_cells, n_timesteps)
            t_arr = np.asarray(t_hist, dtype=np.float64)
            dt_t = t_new - t_arr  # broadcast to shape (n_timesteps,)
            # Vectorized kernel: sqrt(t_new - t_k) - sqrt(t_new - t_k-1)
            sqrt_dt_right = np.sqrt(np.maximum(dt_t[1:], 0.0))
            sqrt_dt_left = np.sqrt(np.maximum(dt_t[:-1], 0.0))
            kernel = sqrt_dt_left - sqrt_dt_right  # shape (n_timesteps-1,)
            history_term = np.dot(q_arr, kernel)  # (n_cells, n_timesteps-1) @ (n_timesteps-1,) = (n_cells,)

        sqrt_dt = np.sqrt(max(dt, 1e-12))
        acoef = coeff * sqrt_dt * htc
        Ts_new = (Tp0 + coeff * history_term + acoef * Tg) / (1.0 + acoef)
        q_new = htc * (Tg - Ts_new)

        thermal_state['Ts'] = Ts_new
        # print(np.max(Tg), np.min(Tg))
        # print(np.max(Ts_new), np.min(Ts_new))
        thermal_state['ignited'] = thermal_state['ignited'] | (Ts_new >= TSurf)
        thermal_state['q_hist'].append(q_new.copy())
        thermal_state['t_hist'].append(t_new)
        thermal_state['time'] = t_new
        return thermal_state

    
    def source_term(U, A, P, ignited, dt, t_local, k):
        _, _, p, _, _ = primitives(U, A, k)

        Ainterface = 0.5 * (A[updated_cell_slice.start:updated_cell_slice.stop+1] + A[updated_cell_slice.start-1:updated_cell_slice.stop])
        dAdx = (Ainterface[1:] - Ainterface[:-1]) / dx  # length nx-6

        S = np.zeros((3, p[updated_cell_slice].shape[0]), dtype=np.float64)
        S[1, :] = p[updated_cell_slice] * dAdx

        rb = np.zeros_like(P[real_cell_slice])

        if erosive == True:
            # rb = arb * p[3:-3]**nrb  # example burning rate r = a*p^n, with a=0.01, n=0.5
            rb = erosive_burning(U, A, P, dt, k)  # compute erosive burning rate based on local flow conditions
            rb = np.where(ignited, rb, 0.0)
            Sburn = P[real_cell_slice]
            S[0, :-1] = rb * Sburn * rhosolid  # mass loss from burning, proportional to pressure and burning area
            S[2, :-1] = rb * Sburn * rhosolid * hreaction  # energy release from burning, proportional to pressure and burning area
            if t_local < 0.35:
                S[0, 3:7] = S[0, 3:7] + mig / (4*dx)
                S[1, 3:7] = S[1, 3:7] + mig * vinj / (4*dx)
                S[2, 3:7] = S[2, 3:7] + mig * hig / (4*dx)

        return S, rb


    def find_dt(U, A, dx, cfl, k):
        llam = max_wave_speed_Toro(U, A, 'dt', k)
        if not np.isfinite(llam):
            return 1e-6
        return cfl * dx / llam if llam > 0 else 1e-6
    
    def SSPRK45(U, A, P, thermal_state, dt, dx, nx, t_local, k, stage_arrays=None):
        # Allocate stage arrays only once per simulation (avoid per-timestep allocation)
        if stage_arrays is None:
            stage_arrays = {
                'U1': np.zeros((3, nx), dtype=np.float64),
                'U2': np.zeros((3, nx), dtype=np.float64),
                'U3': np.zeros((3, nx), dtype=np.float64),
                'U4': np.zeros((3, nx), dtype=np.float64),
            }
        else:
            # Reuse arrays by clearing relevant sections
            for arr in stage_arrays.values():
                arr[:] = 0.0

        U1 = stage_arrays['U1']
        U2 = stage_arrays['U2']
        U3 = stage_arrays['U3']
        U4 = stage_arrays['U4']
        
        inv_dx = -1.0 / dx  # Precompute reciprocal

        thermal_state = update_ignition_thermal_state(U, A, P, thermal_state, dt, k)
        ignited = thermal_state['ignited']

        # Stage 1
        U, k = Riemann_BC(U, A, k)
        U = enforce_positive_state(U, A, k)
        fm, fp = HLLC_flux(U, A, k)
        S1, rb1 = source_term(U, A, P, ignited, dt, t_local, k)
        k1 = inv_dx * (fp - fm) + S1  # Note: sign change absorbed
        U1[:, updated_cell_slice] = U[:, updated_cell_slice] + 0.391752226571890*dt*k1
        U1 = enforce_positive_state(U1, A, k)

        # Stage 2
        U1, k = Riemann_BC(U1, A, k)
        U1 = enforce_positive_state(U1, A, k)
        fm, fp = HLLC_flux(U1, A, k)
        S2, rb2 = source_term(U1, A, P, ignited, dt, t_local, k)
        k2 = inv_dx * (fp - fm) + S2
        U2[:, updated_cell_slice] = 0.444370493651235*U[:, updated_cell_slice] + 0.555629506348765*U1[:, updated_cell_slice] + 0.368410593050371*dt*k2
        U2 = enforce_positive_state(U2, A, k)

        # Stage 3
        U2, k = Riemann_BC(U2, A, k)
        U2 = enforce_positive_state(U2, A, k)
        fm, fp = HLLC_flux(U2, A, k)
        S3, rb3 = source_term(U2, A, P, ignited, dt, t_local, k)
        k3 = inv_dx * (fp - fm) + S3
        U3[:, updated_cell_slice] = 0.620101851488403*U[:, updated_cell_slice] + 0.379898148511597*U2[:, updated_cell_slice] + 0.251891774271694*dt*k3
        U3 = enforce_positive_state(U3, A, k)

        # Stage 4
        U3, k = Riemann_BC(U3, A, k)
        U3 = enforce_positive_state(U3, A, k)
        fm, fp = HLLC_flux(U3, A, k)
        S4, rb4 = source_term(U3, A, P, ignited, dt, t_local, k)
        k4 = inv_dx * (fp - fm) + S4
        U4[:, updated_cell_slice] = 0.178079954393132*U[:, updated_cell_slice] + 0.821920045606868*U3[:, updated_cell_slice] + 0.544974750228521*dt*k4
        U4 = enforce_positive_state(U4, A, k)

        # Stage 5 (final)
        U4, k = Riemann_BC(U4, A, k)
        U4 = enforce_positive_state(U4, A, k)
        fm, fp = HLLC_flux(U4, A, k)
        S5, rb5 = source_term(U4, A, P, ignited, dt, t_local, k)
        k5 = inv_dx * (fp - fm) + S5
        Unp1 = 0.517231671970585*U2[:, updated_cell_slice] + 0.096059710526147*U3[:, updated_cell_slice] + 0.063692468666290*dt*k4 + 0.386708617503268*U4[:, updated_cell_slice] + 0.226007483236906*dt*k5

        # Batch burning rate computation
        rb_cycle = (
            0.1468118760847865 * rb1
            + 0.24848290944497606 * rb2
            + 0.10425883033198079 * rb3
            + 0.27443890090135015 * rb4
            + 0.226007483236906 * rb5
        )

        # Update geometry (vectorized)
        Anew = A.copy()
        Pnew = P.copy()
        Anew[real_cell_slice], Pnew[real_cell_slice] = AP_map(A[real_cell_slice], dt * rb_cycle)
        Anew, Pnew = fill_geometry_ghosts(Anew, Pnew)

        return Unp1, Anew, Pnew, thermal_state, stage_arrays
    
    def step_RK3(U, A, P, thermal_state, dt, dx, nx, t_local, k, stage_arrays=None):
        # Allocate stage arrays only once per simulation (avoid per-timestep allocation)
        if stage_arrays is None:
            stage_arrays = {
                'U1': np.zeros((3, nx), dtype=np.float64),
                'U2': np.zeros((3, nx), dtype=np.float64),
            }
        else:
            for arr in stage_arrays.values():
                arr[:] = 0.0

        U1 = stage_arrays['U1']
        U2 = stage_arrays['U2']
        
        inv_dx = -1.0 / dx  # Precompute reciprocal

        thermal_state = update_ignition_thermal_state(U, A, P, thermal_state, dt, k)
        ignited = thermal_state['ignited']

        # Stage 1
        U, k = Riemann_BC(U, A, k)
        U = enforce_positive_state(U, A, k)
        fm, fp = HLLC_flux(U, A, k)
        S1, rb1 = source_term(U, A, P, ignited, dt, t_local, k)
        k1 = inv_dx * (fp - fm) + S1
        U1[:, updated_cell_slice] = U[:, updated_cell_slice] + dt*k1
        U1 = enforce_positive_state(U1, A, k)

        # Stage 2
        U1, k = Riemann_BC(U1, A, k)
        U1 = enforce_positive_state(U1, A, k)
        fm, fp = HLLC_flux(U1, A, k)
        S2, rb2 = source_term(U1, A, P, ignited, dt, t_local, k)
        k2 = inv_dx * (fp - fm) + S2
        U2[:, updated_cell_slice] = 0.75*U[:, updated_cell_slice] + 0.25*(U1[:, updated_cell_slice] + dt*k2)
        U2 = enforce_positive_state(U2, A, k)

        # Stage 3 (final)
        U2, k = Riemann_BC(U2, A, k)
        U2 = enforce_positive_state(U2, A, k)
        fm, fp = HLLC_flux(U2, A, k)
        S3, rb3 = source_term(U2, A, P, ignited, dt, t_local, k)
        k3 = inv_dx * (fp - fm) + S3
        Unp1 = (1.0/3.0)*U[:, updated_cell_slice] + (2.0/3.0)*(U2[:, updated_cell_slice] + dt*k3)

        rb_cycle = (1.0/6.0)*rb1 + (1.0/6.0)*rb2 + (2.0/3.0)*rb3

        Anew = A.copy()
        Pnew = P.copy()
        Anew[real_cell_slice], Pnew[real_cell_slice] = AP_map(A[real_cell_slice], dt * rb_cycle)
        Anew, Pnew = fill_geometry_ghosts(Anew, Pnew)

        return Unp1, Anew, Pnew, thermal_state, stage_arrays
    
    def plot(Un, A, k):

        # Un = U[:, :, n]

        rho, u, p, E, a = primitives(Un, A, k)

        os.makedirs(f"cfl{cfl}_c{case}_{boundary_case}_t{t}_x{xdom}", exist_ok=True)

        fig, ax = plt.subplots(figsize=(height, width), dpi=dpi)
        # for i in range(0, n+1, max(1, n//10)):
        #     rho_i, u_i, p_i, _, _ = primitives(U, A, i)
        #     ax.plot(rho_i, linewidth=2, label=f"t={t_list[i]:.2e}s")
        ax.plot(xlist, rho, linewidth=4, color=line_color)
        ax.set_ylabel("Densitate rho", fontsize=label_size)
        ax.set_xlabel("x [m]", fontsize=label_size)
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.grid(True)
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)
        ax.set_aspect(ar)
        plt.savefig(f"cfl{cfl}_c{case}_{boundary_case}_t{t}_x{xdom}/rho_end.png")
        plt.clf()

        fig, ax = plt.subplots(figsize=(height, width), dpi=dpi)
        # for i in range(0, n+1, max(1, n//10)):
        #     rho_i, u_i, p_i, _, _ = primitives(U, A, i)
        #     ax.plot(p_i, linewidth=2, label=f"t={t_list[i]:.2e}s")
        ax.plot(xlist, p, linewidth=4, color=line_color)
        ax.set_ylabel("Presiune p [Pa]", fontsize=label_size)
        ax.set_xlabel("x [m]", fontsize=label_size)
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.grid(True)
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)
        ax.set_aspect(ar)
        plt.savefig(f"cfl{cfl}_c{case}_{boundary_case}_t{t}_x{xdom}/p_end.png")
        plt.clf()

        fig, ax = plt.subplots(figsize=(height, width), dpi=dpi)
        # for i in range(0, n+1, max(1, n//10)):
        #     rho_i, u_i, p_i, _, _ = primitives(U, A, i)
        #     ax.plot(u_i, linewidth=2, label=f"t={t_list[i]:.2e}s")
        ax.plot(xlist, u, linewidth=4, color=line_color)
        ax.set_ylabel("Viteza u [m/s]", fontsize=label_size)
        ax.set_xlabel("x [m]", fontsize=label_size)
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.grid(True)
        ax.set_xlim(left=0.0)
        # ax.set_ylim(bottom=0.0)
        ax.set_aspect(ar)
        plt.savefig(f"cfl{cfl}_c{case}_{boundary_case}_t{t}_x{xdom}/u_end.png")
        plt.clf()

        fig, ax = plt.subplots(figsize=(height, width), dpi=dpi)
        # for i in range(0, n+1, max(1, n//10)):
        #     rho_i, u_i, p_i, _, _ = primitives(U, A, i)
        #     ax.plot(u_i, linewidth=2, label=f"t={t_list[i]:.2e}s")
        ax.plot(xlist, u/a, linewidth=4, color=line_color)
        ax.set_ylabel("Numarul Mach M [m/s]", fontsize=label_size)
        ax.set_xlabel("x [m]", fontsize=label_size)
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.grid(True)
        ax.set_xlim(left=0.0)
        # ax.set_ylim(bottom=0.0)
        ax.set_aspect(ar)
        plt.savefig(f"cfl{cfl}_c{case}_{boundary_case}_t{t}_x{xdom}/M_end.png")
        plt.clf()

        fig = plt.figure(figsize=(height, width), dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        t_surface = np.asarray(t_list, dtype=np.float64)
        Tgrid, Xgrid = np.meshgrid(t_surface, xlist, indexing='ij')
        Agrid = A_printlist
        surf = ax.plot_surface(Xgrid, Tgrid, Agrid, cmap='inferno', linewidth=0, antialiased=True)
        fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.1, label='Aria [m^2]')
        ax.set_xlabel("x [m]")
        ax.set_ylabel("Timp [s]")
        ax.set_zlabel("Aria [m^2]")
        ax.tick_params(axis='both', which='major')
        ax.view_init(elev=25, azim=-130)
        plt.savefig(f"cfl{cfl}_c{case}_{boundary_case}_t{t}_x{xdom}/A_end.png")
        # plt.show()
        plt.clf()

        plt.close('all')
    
    # Main simulation loop

    # State
    t = 0.0
    nx = 200
    cfl = 0.9
    erosive = True
    kglobal = kigniter * np.ones(nx, dtype=np.float64)  # ideal gas constant for the igniter gas; can be modified to be spatially varying if needed

    print_progress = True

    # test_cases = np.arange(1, 8)
    test_cases = [38] # for quick testing; comment out to run all cases

    for i in test_cases:
        t = 0.0
        case = i
        left_initial, right_initial, t_end, xdom, boundary_case = test_case(case)
        dx = xdom / (nx - 6)
        xlist = np.linspace(0.0, xdom, nx)  # include ghost cells for BCs

        # Variables to track over time
        U = np.zeros((3, nx, 2000000), dtype=np.float64)  # U[0] = rho*A, U[1] = rho*u*A, U[2] = E*A; shape (nvar, nx, nt) with ghost cells for BCs; will slice to current time step
        Unext = np.zeros((3, nx), dtype=np.float64)  # temporary array for next time step to avoid huge allocations inside SSPRK45
        Ulast = np.zeros((3, nx), dtype=np.float64)  # temporary array for current time step to avoid huge allocations inside SSPRK45
        A = np.ones(nx, dtype=np.float64)  # test cross-sectional area
        P = 2*np.sqrt(np.pi)*np.ones(nx, dtype=np.float64)
        # if nx % 2 == 0:
        #     # A = np.concatenate((1*np.ones(nx//2), 2*np.ones(nx//2)))  # add ghost cells for BCs
        #     A = np.sin(np.linspace(0, np.pi/2, nx)) + 1  # example variable area for testing; shift up to ensure positivity
        # else:
        #     # A = np.concatenate((1*np.ones(nx//2), 2*np.ones(nx//2+1)))  # add ghost cells for BCs 
        #     A = np.sin(np.linspace(0, np.pi/2, nx)) + 1
        A, P = fill_geometry_ghosts(A, P)
        thermal_state = init_solid_thermal_state(nx)
        A_printlist = np.array([A])  # store initial geometry for plotting; will be updated in source_term if erosive=True
        P_printlist = np.array([P])  # store initial geometry for plotting; will be updated in source_term if erosive=True

        p_list = np.zeros((nx, 2000000,), dtype=np.float64)  # preallocate pressure list for plotting; will slice to current time step
        u_list = np.zeros((nx, 2000000,), dtype=np.float64)  # preallocate velocity list for plotting; will slice to current time step
        T_list = np.zeros((nx, 2000000,), dtype=np.float64)  # preallocate temperature list for plotting; will slice to current time step
        rho_list = np.zeros((nx, 2000000,), dtype=np.float64)  # preallocate density list for plotting; will slice to current time step

        # Outputs
        t_list = [t]
        
        n = 0
        U = initial_Riemann(U, A, left_initial, right_initial, kglobal)
        Ulast = U[:, :, 0]
        Ulast, kglobal = Riemann_BC(Ulast, A, kglobal)
        Ulast = enforce_positive_state(Ulast, A, kglobal)

        # Preallocate stage arrays once for all time steps (critical optimization)
        stage_arrays = None

        while t < t_end:
            dt = find_dt(Ulast, A, dx, cfl, kglobal)
            Unext[:, updated_cell_slice], A, P, thermal_state, stage_arrays = SSPRK45(Ulast, A, P, thermal_state, dt, dx, nx, t, kglobal, stage_arrays)
            kglobal[real_cell_slice] = np.where(thermal_state['ignited'], kgas, kigniter)
            kglobal[fictitious_cell_idx:] = kglobal[fictitious_cell_idx - 1]
            Unext, kglobal = Riemann_BC(Unext, A, kglobal)
            Unext = enforce_positive_state(Unext, A, kglobal)
            Ulast = Unext
            t += dt
            n += 1

            # Store outputs
            A_printlist = np.append(A_printlist, A.copy())
            P_printlist = np.append(P_printlist, P.copy())
            A_printlist = np.reshape(A_printlist, (-1, nx))
            P_printlist = np.reshape(P_printlist, (-1, nx))

            t_list.append(t)
            if print_progress:
                print(t)
            if n % 1000 == 0:
                plot(Ulast, A, kglobal)
            
        A_printlist = np.reshape(A_printlist, (-1, nx))
        P_printlist = np.reshape(P_printlist, (-1, nx))

        t_list = np.asarray(t_list, dtype=np.float64)
        U = U[:, :, :n+1]
        plot(Ulast, A, kglobal)
        # plt.show()


if __name__ == "__main__":
    simulate()
