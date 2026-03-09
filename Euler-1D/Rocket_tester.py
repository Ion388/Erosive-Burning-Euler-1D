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
import math
import os
import sys
from Riemann_test_cases import test_case

# Hint BLAS/OpenMP backends to use all CPU threads for vectorized kernels.
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

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
    
    def primitives(U, A):
        """Convert conserved variables U to primitive variables (rho, u, p, E, a).
        U = [rho*A, rho*u*A, rho*E*A], where E is total energy per volume. A is cross-sectional area.
        """

        if U.shape[1] != A.shape[0]:
            Ause = A[2:-2]  # trim A to match U's spatial dimension if needed
            Ause = 0.5*(Ause[:-1] + Ause[1:])  # average adjacent A values to get A at cell centers for better accuracy in primitives
        else:
            Ause = A

        rho = U[0, :] / Ause
        u   = U[1, :] / (rho * Ause)
        E   = U[2, :] / (rho * Ause)    # total specific energy (per unit mass)
        
        p = rho * (k - 1.0) * (E - 0.5 * u*u)
        a = np.sqrt(k * p / rho)
        return rho, u, p, E, a
    
    def initial_Riemann(U, A, left_initial, right_initial):
        shape = U.shape
        Aext = np.zeros([3, shape[1]], dtype=np.float64)
        Aext = np.tile(A, (3, 1))  # extend A to match U's spatial dimension if needed

        rhoL = left_initial[0]
        pL = left_initial[1]
        uL = left_initial[2]
        left_state = np.array([rhoL, rhoL*uL, (pL/(k-1) + 0.5*rhoL*uL*uL)])  # (rho*A, rho*u*A, rho*E*A) left state

        rhoR = right_initial[0]
        pR = right_initial[1]
        uR = right_initial[2]
        right_state = np.array([rhoR, rhoR*uR, (pR/(k-1) + 0.5*rhoR*uR*uR)])  # (rho*A, rho*u*A, rho*E*A) right state
        
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


    def Riemann_BC(U, A):

        if boundary_case == 'Riemann-wall':
            left_dst = np.array([2, 1, 0])
            left_src = np.array([3, 4, 5])
            U[:, left_dst] = U[:, left_src]
            right_dst = np.array([-3, -2, -1])
            right_src = np.array([-4, -5, -6])
            U[:, right_dst] = U[:, right_src]
            U[1, right_dst] = -U[1, right_dst]  # reflective wall on the right side
            return U

        if boundary_case == 'Riemann':
            left_dst = np.array([2, 1, 0])
            left_src = np.array([3, 4, 5])
            U[:, left_dst] = U[:, left_src]
            right_dst = np.array([-3, -2, -1])
            right_src = np.array([-4, -5, -6])
            U[:, right_dst] = U[:, right_src]
            return U

        # Left boundary: rigid reflective wall.
        left_dst = np.array([2, 1, 0])
        left_src = np.array([3, 4, 5])
        U[:, left_dst] = U[:, left_src]
        U[1, left_dst] = -U[1, left_src]


        if boundary_case == 'wall-wall':
            # Right boundary: rigid reflective wall.
            right_dst = np.array([-3, -2, -1])
            right_src = np.array([-4, -5, -6])
            U[:, right_dst] = U[:, right_src]
            U[1, right_dst] = -U[1, right_src]
            return U

        if boundary_case != 'wall-atmosphere':
            raise ValueError("boundary_case must be 'Riemann-wall' or 'wall-wall' or 'wall-atmosphere'.")

        # Right boundary: atmospheric outlet/inlet.
        # - Supersonic outflow: all characteristics leave, use zero-gradient.
        # - Subsonic outflow: impose ambient pressure, extrapolate rho and u.
        # - Backflow: impose full ambient state.
        i_in = -4
        rho_in = U[0, i_in] / A[i_in]  # avoid division by zero and negative density
        u_in = U[1, i_in] / (rho_in * A[i_in])
        E_in = U[2, i_in] / (rho_in * A[i_in])
        p_in = rho_in * (k - 1.0) * (E_in - 0.5 * u_in * u_in)
        a_in = math.sqrt(k * p_in / rho_in)

        if u_in >= a_in:
            # Supersonic outflow: copy nearest interior state.
            U[:, -3:] = U[:, i_in:i_in+1]
            return U

        if u_in >= 0.0:
            # Subsonic outflow: ambient pressure closure.
            rho_g = rho_in
            u_g = u_in
            p_g = p0
        else:
            # Backflow from atmosphere.
            rho_g = rho0
            u_g = 0.0
            p_g = p0

        E_g = p_g / (k - 1.0) + 0.5 * rho_g * u_g * u_g
        U[0, -3:] = rho_g * A[-3:]  # assume area is constant at the boundary for simplicity; could also extrapolate A if needed
        U[1, -3:] = rho_g * u_g * A[-3:]
        U[2, -3:] = E_g * A[-3:]
        return U
    
    def weno5_reconstruct(U, A):
        """
        Characteristic WENO5 reconstruction for 1D Euler variables.

        U shape: (nvar, nx), with at least 3 ghost cells on each side.
        Returns arrays with shape (nvar, nx-6), for i = 3..nx-4:
        - UmL: U^- at x_{i-1/2}
        - UmR: U^+ at x_{i-1/2}
        - UpL: U^- at x_{i+1/2}
        - UpR: U^+ at x_{i+1/2}
        """
        nvar, nx = U.shape
        if nx < 7:
            raise ValueError("Need at least 7 points (including ghost cells) for WENO5.")
        if nvar != 3:
            raise ValueError("Characteristic WENO5 here is implemented for 1D Euler with 3 conserved variables.")

        eps_weno = 1e-10

        def eno3_left_iphalf(w_im2, w_im1, w_i, w_ip1, w_ip2):
            p0 = (1.0 / 3.0) * w_im2 - (7.0 / 6.0) * w_im1 + (11.0 / 6.0) * w_i
            p1 = -(1.0 / 6.0) * w_im1 + (5.0 / 6.0) * w_i + (1.0 / 3.0) * w_ip1
            p2 = (1.0 / 3.0) * w_i + (5.0 / 6.0) * w_ip1 - (1.0 / 6.0) * w_ip2

            gamma1 = 1/10
            gamma2 = 3/5
            gamma3 = 3/10   

            beta1 = (13/12)*(w_im2 - 2*w_im1 + w_i)**2 + (1/4)*(w_im2 - 4*w_im1 + 3*w_i)**2
            beta2 = (13/12)*(w_im1 - 2*w_i + w_ip1)**2 + (1/4)*(w_im1 - w_ip1)**2
            beta3 = (13/12)*(w_i - 2*w_ip1 + w_ip2)**2 + (1/4)*(3*w_i - 4*w_ip1 + w_ip2)**2

            alpha1 = gamma1/(eps_weno + beta1)**2
            alpha2 = gamma2/(eps_weno + beta2)**2
            alpha3 = gamma3/(eps_weno + beta3)**2

            w1 = alpha1/(alpha1+alpha2+alpha3)
            w2 = alpha2/(alpha1+alpha2+alpha3)
            w3 = alpha3/(alpha1+alpha2+alpha3)

            return w1*p0 + w2*p1 + w3*p2

        def eno3_right_iphalf(w_im1, w_i, w_ip1, w_ip2, w_ip3):
            p0 = -(1.0 / 6.0) * w_im1 + (5.0 / 6.0) * w_i + (1.0 / 3.0) * w_ip1
            p1 = (1.0 / 3.0) * w_i + (5.0 / 6.0) * w_ip1 - (1.0 / 6.0) * w_ip2
            p2 = (11.0 / 6.0) * w_ip1 - (7.0 / 6.0) * w_ip2 + (1.0 / 3.0) * w_ip3

            gamma1 = 3/10
            gamma2 = 3/5
            gamma3 = 1/10

            beta1 = (13/12)*(w_im1 - 2*w_i + w_ip1)**2 + (1/4)*(w_im1 - 4*w_i + 3*w_ip1)**2
            beta2 = (13/12)*(w_i - 2*w_ip1 + w_ip2)**2 + (1/4)*(w_i - w_ip2)**2
            beta3 = (13/12)*(w_ip1 - 2*w_ip2 + w_ip3)**2 + (1/4)*(3*w_ip1 - 4*w_ip2 + w_ip3)**2

            alpha1 = gamma1/(eps_weno + beta1)**2
            alpha2 = gamma2/(eps_weno + beta2)**2
            alpha3 = gamma3/(eps_weno + beta3)**2

            w1 = alpha1/(alpha1+alpha2+alpha3)
            w2 = alpha2/(alpha1+alpha2+alpha3)
            w3 = alpha3/(alpha1+alpha2+alpha3)

            return w1*p0 + w2*p1 + w3*p2

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

        # rhoL = np.maximum(u_im1[0], eps)
        rhoL = u_im1[0] / A_im1
        velL = u_im1[1] / (rhoL * A_im1)
        EL = u_im1[2] / (rhoL * A_im1)
        # pL = np.maximum(rhoL * (k - 1.0) * (EL - 0.5 * velL * velL), eps)
        pL = rhoL * (k - 1.0) * (EL - 0.5 * velL * velL)
        HL = EL + pL / rhoL

        # rhoR = np.maximum(u_i[0], eps)
        rhoR = u_i[0] / A_i
        velR = u_i[1] / (rhoR * A_i)
        ER = u_i[2] / (rhoR * A_i)
        # pR = np.maximum(rhoR * (k - 1.0) * (ER - 0.5 * velR * velR), eps)
        pR = rhoR * (k - 1.0) * (ER - 0.5 * velR * velR)
        HR = ER + pR / rhoR

        sL = np.sqrt(rhoL)
        sR = np.sqrt(rhoR)
        # denom = np.maximum(sL + sR, eps)
        denom = sL + sR

        rhohat = sL * sR
        uhat = (sL * velL + sR * velR) / denom
        Hhat = (sL * HL + sR * HR) / denom
        # ahat2 = np.maximum((k - 1.0) * (Hhat - 0.5 * uhat * uhat), eps)
        ahat2 = (k - 1.0) * (Hhat - 0.5 * uhat * uhat)
        ahat = np.sqrt(ahat2)

        P = np.empty((m, 3, 3), dtype=U.dtype)
        P[:, 0, 0] = 1.0
        P[:, 0, 1] = rhohat / (2.0 * ahat)
        P[:, 0, 2] = rhohat / (2.0 * ahat)
        P[:, 1, 0] = uhat
        P[:, 1, 1] = rhohat * (uhat + ahat) / (2.0 * ahat)
        P[:, 1, 2] = rhohat * (uhat - ahat) / (2.0 * ahat)
        P[:, 2, 0] = 0.5 * uhat * uhat
        P[:, 2, 1] = rhohat * (Hhat + uhat * ahat) / (2.0 * ahat)
        P[:, 2, 2] = rhohat * (Hhat - uhat * ahat) / (2.0 * ahat)

        L = np.linalg.inv(P)
        eigenvals = np.vstack((uhat, uhat + ahat, uhat - ahat))

        wm_im3 = np.einsum('mab,bm->am', L, u_im3)
        wm_im2 = np.einsum('mab,bm->am', L, u_im2)
        wm_im1 = np.einsum('mab,bm->am', L, u_im1)
        wm_i = np.einsum('mab,bm->am', L, u_i)
        wm_ip1 = np.einsum('mab,bm->am', L, u_ip1)
        wm_ip2 = np.einsum('mab,bm->am', L, u_ip2)

        w_uL = eno3_left_iphalf(wm_im3, wm_im2, wm_im1, wm_i, wm_ip1)
        w_uR = eno3_right_iphalf(wm_im2, wm_im1, wm_i, wm_ip1, wm_ip2)
        UL = np.einsum('mab,bm->am', P, w_uL)
        UR = np.einsum('mab,bm->am', P, w_uR)
        return UL, UR, eigenvals
    
    # Wave speed
    def max_wave_speed_Toro(U, A, case):
        UL, UR, _ = weno5_reconstruct(U, A)

        rhoL, uL, pL, _, aL = primitives(UL, A)  # primitives at x_{i-1/2} left state
        rhoR, uR, pR, _, aR = primitives(UR, A)  # primitives at x_{i-1/2} right state

        gamma_exp = (k - 1) / (2 * k)
        power_exp = 2 * k / (k - 1)
        base_num = aL + aR - 0.5 * (k - 1) * (uR - uL)
        base_den = aL / (pL ** gamma_exp) + aR / (pR ** gamma_exp)
        base = base_num / base_den
        pstarr = np.power(base, power_exp)  # Toro's p* estimate at x_{i-1/2}
        #### ALTERNATE METHOD FOR p* ####
        # rhohat = 0.5 * (rhoL + rhoR)
        # ahat = 0.5 * (aL + aR)
        # ppvrs = 0.5 * (pL + pR) - 0.5 * (uR - uL) * rhohat * ahat
        # pstarr = np.maximum(0.0, ppvrs)  # Toro's p* estimate at x_{i-1/2}, with positivity preservation

        qL = np.where(pstarr <= pL, 1.0, np.sqrt(1.0 + (k + 1.0) / (2.0 * k) * (pstarr / pL - 1.0)))
        qR = np.where(pstarr <= pR, 1.0, np.sqrt(1.0 + (k + 1.0) / (2.0 * k) * (pstarr / pR - 1.0)))

        SL = uL - aL*qL
        SR = uR + aR*qR
        
        if case == 'dt':
            return np.max(np.abs([SL, SR]))
        elif case == 'flux':
            return SL, SR

    def Euler_flux(U, A):
        if U.shape[1] != A.shape[0]:
            Ause = A[2:-2]  # trim A to match U's spatial dimension if needed
            Ause = 0.5*(Ause[:-1] + Ause[1:])  # Area at the interfaces, averaged for better accuracy in flux calculation
        else:
            Ause = A
        rho, u, p, E, a = primitives(U, A)
        return np.vstack((rho*u*Ause, (rho*u**2 + p)*Ause, u*(rho*E + p)*Ause))

    def HLLC_flux(U, A):
        UL, UR, eigenvals = weno5_reconstruct(U, A)
        rhoL, uL, pL, EL, aL = primitives(UL, A)  # primitives at left state
        rhoR, uR, pR, ER, aR = primitives(UR, A)  # primitives at right state

        fL = Euler_flux(UL, A)
        fR = Euler_flux(UR, A)

        Ause = A[2:-2]
        Ause = 0.5*(Ause[:-1] + Ause[1:])

        # if wave_speed_method == 'Dava':
        #     SL = np.minimum(np.min(eigenvals, axis=0), 0)  # Davidson's HLL wave speed estimates
        #     SR = np.maximum(np.max(eigenvals, axis=0), 0)  # Davidson's HLL wave speed estimates
        # elif wave_speed_method == 'Toro':
        SL, SR = max_wave_speed_Toro(U, A, case='flux')  # Toro's max wave speeds at x_{i-1/2} and x_{i+1/2}

        Sstar = (pR - pL + rhoL*uL*(SL-uL) - rhoR*uR*(SR-uR))/(rhoL*(SL-uL) - rhoR*(SR-uR))

        UstarL = Ause * rhoL * (SL - uL) / (SL - Sstar) * np.array([np.ones(SL.shape), Sstar, EL + (Sstar - uL)*(Sstar + pL/(rhoL*(SL-uL)))])
        UstarR = Ause * rhoR * (SR - uR) / (SR - Sstar) * np.array([np.ones(SR.shape), Sstar, ER + (Sstar - uR)*(Sstar + pR/(rhoR*(SR-uR)))])
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

        return fm, fp

    def AP_map(A0, dtrb):
        r0 = np.sqrt(A0/np.pi)
        A1 = np.pi * (r0 + dtrb)**2
        P1 = 2 * np.sqrt(A1 * np.pi)
        return A1, P1

    def fill_geometry_ghosts(A, P):
        # linear extrapolation for better accuracylinear extrapolation for better accuracy
        A[2] = A[3] - (A[4] - A[3])
        A[1] = A[3] - 2*(A[4] - A[3])
        A[0] = A[3] - 3*(A[4] - A[3])

        A[-3] = A[-4] - (A[-5] - A[-4])
        A[-2] = A[-4] - 2*(A[-5] - A[-4])
        A[-1] = A[-4] - 3*(A[-5] - A[-4])

        P[2] = P[3] - (P[4] - P[3])
        P[1] = P[3] - 2*(P[4] - P[3])
        P[0] = P[3] - 3*(P[4] - P[3])

        P[-3] = P[-4] - (P[-5] - P[-4])
        P[-2] = P[-4] - 2*(P[-5] - P[-4])
        P[-1] = P[-4] - 3*(P[-5] - P[-4])

        return A, P

    
    def source_term(U, A, P, dt):
        _, _, p, _, _ = primitives(U, A)

        Ainterface = 0.5 * (A[3:-2] + A[2:-3])
        dAdx = (Ainterface[1:] - Ainterface[:-1]) / dx  # length nx-6

        S = np.zeros((3, p[3:-3].shape[0]), dtype=np.float64)
        S[1, :] = p[3:-3] * dAdx

        A1 = A.copy()
        P1 = P.copy()

        if erosive == True:
            rb = arb * p[3:-3]**nrb  # example burning rate r = a*p^n, with a=0.01, n=0.5
            Sburn = dx * P[3:-3]
            S[0, :] = rb * Sburn * rhoprop  # mass loss from burning, proportional to pressure and burning area
            S[2, :] = rb * Sburn * rhoprop * hreaction  # energy release from burning, proportional to pressure and burning area
            A1[3:-3], P1[3:-3] = AP_map(A[3:-3], dt*rb)  # update interior geometry
            A1, P1 = fill_geometry_ghosts(A1, P1)
        
        return S, A1, P1


    def find_dt(U, A, dx, cfl):
        llam = max_wave_speed_Toro(U, A, case='dt')
        return cfl * dx / llam if llam > 0 else 1e-6
    
    def SSPRK45(U, A, P, dt, dx, nx):

        # Keep only one stage index to avoid huge allocations each time step.
        U1 = np.zeros((3, nx), dtype=np.float64)
        U2 = np.zeros((3, nx), dtype=np.float64)
        U3 = np.zeros((3, nx), dtype=np.float64)
        U4 = np.zeros((3, nx), dtype=np.float64)

        # shock reflection NOT depicted accurately for now due to sboundary conditions
        U = Riemann_BC(U, A)
        fm, fp = HLLC_flux(U, A)
        S1, A1, P1 = source_term(U, A, P, dt)
        k1 = -1/dx * (fp - fm) + S1
        U1[:, 3:nx-3] = U[:, 3:nx-3] + 0.391752226571890*dt*k1

        U1 = Riemann_BC(U1, A1)
        fm, fp = HLLC_flux(U1, A1)
        S2, A2, P2 = source_term(U1, A1, P1, dt)
        k2 = -1/dx * (fp - fm) + S2
        U2[:, 3:nx-3] = 0.444370493651235*U[:, 3:nx-3] + 0.555629506348765*U1[:, 3:nx-3] + 0.368410593050371*dt*k2

        U2 = Riemann_BC(U2, A2)
        fm, fp = HLLC_flux(U2, A2)
        S3, A3, P3 = source_term(U2, A2, P2, dt)
        k3 = -1/dx * (fp - fm) + S3
        U3[:, 3:nx-3] = 0.620101851488403*U[:, 3:nx-3] + 0.379898148511597*U2[:, 3:nx-3] + 0.251891774271694*dt*k3

        U3 = Riemann_BC(U3, A3)
        fm, fp = HLLC_flux(U3, A3)
        S4, A4, P4 = source_term(U3, A3, P3, dt)
        k4 = -1/dx * (fp - fm) + S4
        U4[:, 3:nx-3] = 0.178079954393132*U[:, 3:nx-3] + 0.821920045606868*U3[:, 3:nx-3] +  0.544974750228521*dt*k4

        U4 = Riemann_BC(U4, A4)
        fm, fp = HLLC_flux(U4, A4)
        S5, A5, P5 = source_term(U4, A4, P4, dt)
        k5 = -1/dx * (fp - fm) + S5
        Unp1 = 0.517231671970585*U2[:, 3:nx-3] + 0.096059710526147*U3[:, 3:nx-3] + 0.063692468666290*dt*k4 + 0.386708617503268*U4[:, 3:nx-3] + 0.226007483236906*dt*k5

        return Unp1, A5, P5
    
    def plot(U, A, n):

        Un = U[:, :, n]

        rho, u, p, E, a = primitives(Un, A)

        os.makedirs(f"cfl{cfl}_c{case}_{boundary_case}_t{t_end}_x{xdom}", exist_ok=True)

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
        plt.savefig(f"cfl{cfl}_c{case}_{boundary_case}_t{t_end}_x{xdom}/rho_end.png")
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
        plt.savefig(f"cfl{cfl}_c{case}_{boundary_case}_t{t_end}_x{xdom}/p_end.png")
        # plt.clf()

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
        plt.savefig(f"cfl{cfl}_c{case}_{boundary_case}_t{t_end}_x{xdom}/u_end.png")
        # plt.clf()

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
        plt.savefig(f"cfl{cfl}_c{case}_{boundary_case}_t{t_end}_x{xdom}/M_end.png")
        # plt.clf()

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
        plt.savefig(f"cfl{cfl}_c{case}_{boundary_case}_t{t_end}_x{xdom}/A_end.png")
        # plt.show()
        # plt.clf()
    
    # Main simulation loop

    k = 1.4
    Rgas = 287  # J/kg-K

    # State
    t = 0.0
    nx = 1001
    cfl = 0.9
    arb = 0.01  # burning rate coefficient for source term example
    nrb = 0.5   # burning rate exponent for source term example
    rhoprop = 1800  # propellant density for source term example
    hreaction = 3e6  # reaction heat release for source term example
    erosive = False

    print_progress = True


    

    # Atmospheric conditions used when boundary_case = 'wall-atmosphere'.
    # p0 = 100000  # Pa
    # rho0 = 1.000   # kg/m^3
    # T0 = 348    # K
    # p0_check = rho0 * Rgas * T0
    # if abs(p0_check - p0) / p0 > 5e-2:
    #     print(f"Warning: p0, rho0, T0 are not fully consistent with ideal gas: rho0*R*T={p0_check:.2f} Pa")

    test_cases = np.arange(1, 8)
    # test_cases = [8] # for quick testing; comment out to run all cases

    for i in test_cases:
        t = 0.0
        case = i
        left_initial, right_initial, t_end, xdom, boundary_case = test_case(case)
        dx = xdom / (nx - 6)
        xlist = np.linspace(0.0, xdom, nx)  # include ghost cells for BCs

        # Variables to track over time
        U = np.zeros((3, nx, 200000), dtype=np.float64)  # U[0] = rho*A, U[1] = rho*u*A, U[2] = E*A; shape (nvar, nx, nt) with ghost cells for BCs; will slice to current time step
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
        A_printlist = np.array([A])  # store initial geometry for plotting; will be updated in source_term if erosive=True
        P_printlist = np.array([P])  # store initial geometry for plotting; will be updated in source_term if erosive=True

        # Outputs
        t_list = [t]
        
        n = 0
        U = initial_Riemann(U, A, left_initial, right_initial)
        U[:, :, 0] = Riemann_BC(U[:, :, 0], A)

        while t < t_end:
            Ulast = U[:, :, n]
            dt = find_dt(Ulast, A, dx, cfl)
            Unext[:, 3:nx-3], A, P = SSPRK45(Ulast, A, P, dt, dx, nx)
            Unext = Riemann_BC(Unext, A)  # Apply BCs to the new state before storing it
            U[:, :, n+1] = Unext
            t += dt
            n += 1

            # Store outputs
            A_printlist = np.append(A_printlist, A.copy())  # store geometry at each time step for plotting; will be updated in source_term if erosive=True
            P_printlist = np.append(P_printlist, P.copy())  # store geometry at each time step for plotting; will be updated in source_term if erosive=True   
            t_list.append(t)
            if print_progress:
                print(t)
        A_printlist = np.reshape(A_printlist, (-1, nx))
        P_printlist = np.reshape(P_printlist, (-1, nx))

        t_list = np.asarray(t_list, dtype=np.float64)
        U = U[:, :, :n+1]
        plot(U, A, n)
        # plt.show()


if __name__ == "__main__":
    simulate()
