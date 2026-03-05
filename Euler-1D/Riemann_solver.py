### DE RESCRIS INFO DESPRE PROGRAM ###
# Notes:
# - All units SI. Time in seconds, Pa, m, kg.
# 1 = chamber
# 2 = nozzle exit
# 3/t = throat
# amb = ambient

from __future__ import annotations
import math
import os
import sys
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
    
    def primitives(U, A, n):
        """Convert conserved variables U to primitive variables (rho, u, p, E, a).
        U = [rho*A, rho*u*A, rho*E*A], where E is total energy per volume. A is cross-sectional area.
        """
        A_place = 1
        if U.ndim == 3:
            rho = U[0, :, n]
            u   = U[1, :, n] / rho
            E   = U[2, :, n] / rho     # total specific energy (per unit mass)
        else:
            rho = U[0, :]
            u   = U[1, :] / rho
            E   = U[2, :] / rho    # total specific energy (per unit mass)
        
        p = rho * (k - 1.0) * (E - 0.5 * u*u)
        a = np.sqrt(k * p / rho)
        return rho, u, p, E, a
    
    def initial_Riemann(U, A, n):
        shape = U.shape

        rhoL = 1.0
        pL = 1.0e5
        uL = 0.0
        left_state = np.array([rhoL, rhoL*uL, (pL/(k-1) + 0.5*rhoL*uL*uL)])  # (rho*A, rho*u*A, rho*E*A) left state

        rhoR = 0.125
        pR = 1.0e4
        uR = 0.0
        right_state = np.array([rhoR, rhoR*uR, (pR/(k-1) + 0.5*rhoR*uR*uR)])  # (rho*A, rho*u*A, rho*E*A) right state
        
        UL = np.repeat(left_state, shape[1]//2, axis=0).reshape(3, shape[1]//2)
        UR = np.repeat(right_state, shape[1]//2, axis=0).reshape(3, shape[1]//2)

        # even number of points necessary!
        U[:, :shape[1]//2, n] = UL
        U[:, shape[1]//2:, n] = UR
        return U


    def Riemann_BC(U, n):
        Ulast = U[:, :, n]
        # Simple reflective BCs for now (2 ghost cells on each side)
        Ulast[:, 0] = Ulast[:, 3]  # left ghost cell 1
        Ulast[:, 1] = Ulast[:, 3]  # left ghost cell 2 (can be same as first for simplicity)
        Ulast[:, 2] = Ulast[:, 3]  # left ghost cell 3 (can be same as first for simplicity)
        Ulast[:, -1] = Ulast[:, -4]  # right ghost cell 1
        Ulast[:, -2] = Ulast[:, -4]  # right ghost cell 2 (can be same as first for simplicity)
        Ulast[:, -3] = Ulast[:, -4]  # right ghost cell 3 (can be same as first for simplicity)

        # speed is reflected, not symmetric
        # u   = Ulast[1] / Ulast[0]
        Ulast[1, 0] = -Ulast[1, 3]  # left ghost cell 1
        Ulast[1, 1] = -Ulast[1, 3]  # left ghost cell 2 (can be same as first for simplicity)
        Ulast[1, 2] = -Ulast[1, 3]  # left ghost cell 3 (can be same as first for simplicity)
        Ulast[1, -1] = -Ulast[1, -4]  # right ghost cell 1
        Ulast[1, -2] = -Ulast[1, -4]  # right ghost cell 2 (can be same as first for simplicity)
        Ulast[1, -3] = -Ulast[1, -4]  # right ghost cell 3 (can be same as first for simplicity)
        U[:, :, n] = Ulast  # update U with new BCs for current time step
        return U
    
    def weno5_reconstruct(U, n):
        """
        Characteristic WENO5 reconstruction for 1D Euler variables.

        U shape: (nvar, nx, nt), with at least 3 ghost cells on each side.
        Returns arrays with shape (nvar, nx-6), for i = 3..nx-4:
        - UmL: U^- at x_{i-1/2}
        - UmR: U^+ at x_{i-1/2}
        - UpL: U^- at x_{i+1/2}
        - UpR: U^+ at x_{i+1/2}
        """
        nvar, nx, nt = U.shape
        Ulast = U[:, :, n]  # current time step
        if nx < 7:
            raise ValueError("Need at least 7 points (including ghost cells) for WENO5.")
        if nvar != 3:
            raise ValueError("Characteristic WENO5 here is implemented for 1D Euler with 3 conserved variables.")

        eps = 1e-12

        def eig_lr(UL: np.ndarray, UR: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            rhoL = max(UL[0], eps)
            uL = UL[1] / rhoL
            EL = UL[2] / rhoL
            pL = max(rhoL * (k - 1.0) * (EL - 0.5 * uL * uL), eps)
            HL = EL + pL / rhoL

            rhoR = max(UR[0], eps)
            uR = UR[1] / rhoR
            ER = UR[2] / rhoR
            pR = max(rhoR * (k - 1.0) * (ER - 0.5 * uR * uR), eps)
            HR = ER + pR / rhoR

            sL = math.sqrt(rhoL)
            sR = math.sqrt(rhoR)
            denom = max(sL + sR, eps)

            rhohat = sL * sR
            uhat = (sL * uL + sR * uR) / denom
            Hhat = (sL * HL + sR * HR) / denom
            ahat2 = max((k - 1.0) * (Hhat - 0.5 * uhat * uhat), eps)
            ahat = math.sqrt(ahat2)

            Pmatrix = np.array([[1, rhohat/(2*ahat), rhohat/(2*ahat)],
                                [uhat, rhohat*(uhat+ahat)/(2*ahat), rhohat*(uhat-ahat)/(2*ahat)],
                                [uhat*uhat/2, rhohat*(Hhat+uhat*ahat)/(2*ahat), rhohat*(Hhat-uhat*ahat)/(2*ahat)]])

            eigenvalues = np.array([uhat, uhat + ahat, uhat - ahat])
            Pmatrix_inverse = np.linalg.inv(Pmatrix)
            return Pmatrix_inverse, Pmatrix, eigenvalues

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

            eps_weno = 1e-6
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

            beta1 = (13/12)*(w_im1 - 2*w_i + w_ip1)**2 + (1/4)*(3*w_im1 - 4*w_i + w_ip1)**2
            beta2 = (13/12)*(w_i - 2*w_ip1 + w_ip2)**2 + (1/4)*(w_i - w_ip2)**2
            beta3 = (13/12)*(w_ip1 - 2*w_ip2 + w_ip3)**2 + (1/4)*(w_ip1 - 4*w_ip2 + 3*w_ip3)**2

            eps_weno = 1e-6
            alpha1 = gamma1/(eps_weno + beta1)**2
            alpha2 = gamma2/(eps_weno + beta2)**2
            alpha3 = gamma3/(eps_weno + beta3)**2

            w1 = alpha1/(alpha1+alpha2+alpha3)
            w2 = alpha2/(alpha1+alpha2+alpha3)
            w3 = alpha3/(alpha1+alpha2+alpha3)

            return w1*p0 + w2*p1 + w3*p2

        nc = nx - 6
        UL = np.zeros((nvar, nc+1), dtype=Ulast.dtype)
        UR = np.zeros((nvar, nc+1), dtype=Ulast.dtype)
        eigenvals = np.zeros((nvar, nc+1), dtype=Ulast.dtype)

        for j, ic in enumerate(range(3, nx - 2)):
            u_im3 = Ulast[:, ic - 3]
            u_im2 = Ulast[:, ic - 2]
            u_im1 = Ulast[:, ic - 1]
            u_i = Ulast[:, ic]
            u_ip1 = Ulast[:, ic + 1]
            u_ip2 = Ulast[:, ic + 2]

            L, R, eigenvals_j = eig_lr(u_im1, u_i)

            # Project stencil points to characteristic space (minus and plus interfaces)
            wm_im3 = L @ u_im3
            wm_im2 = L @ u_im2
            wm_im1 = L @ u_im1
            wm_i = L @ u_i
            wm_ip1 = L @ u_ip1
            wm_ip2 = L @ u_ip2


            w_uL = np.zeros(nvar, dtype=Ulast.dtype)
            w_uR = np.zeros(nvar, dtype=Ulast.dtype)

            for pidx in range(nvar):
                # UmL: U^- at x_{i-1/2} = left ENO3 at (i-1)+1/2
                w_uL[pidx] = eno3_left_iphalf(
                    wm_im3[pidx], wm_im2[pidx], wm_im1[pidx], wm_i[pidx], wm_ip1[pidx]
                )
                # UmR: U^+ at x_{i-1/2} = right ENO3 at (i-1)+1/2
                w_uR[pidx] = eno3_right_iphalf(
                    wm_im2[pidx], wm_im1[pidx], wm_i[pidx], wm_ip1[pidx], wm_ip2[pidx]
                )


            UL[:, j] = R @ w_uL
            UR[:, j] = R @ w_uR 
            eigenvals[:, j] = eigenvals_j

        return UL, UR, eigenvals
    
    # Wave speed
    # Option 1 (very simple): max local characteristic speed from primitives
    def max_wave_speed_Dava(U, A, n, case):
        UL, UR, _ = weno5_reconstruct(U, n)
        _, uL, _, _, aL = primitives(UL, A, n)
        _, uR, _, _, aR = primitives(UR, A, n)
        SL = uL - aL
        SR = uR + aR

        if case == 'dt':
            return np.max(np.abs([SL, SR]))
        elif case == 'flux':
            return SL, SR
    
    def max_wave_speed_Davb(U, A, n, case):
        UL, UR, _ = weno5_reconstruct(U, n)
        _, uL, _, _, aL = primitives(UL, A, n)
        _, uR, _, _, aR = primitives(UR, A, n)
        SL = np.min([uL - aL, uR - aR], axis=0)
        SR = np.max([uL + aL, uR + aR], axis=0)

        if case == 'dt':
            return np.max(np.abs([SL, SR]))
        elif case == 'flux':
            return SL, SR

    def max_wave_speed_Toro(U, A, n, case):
        UL, UR, _ = weno5_reconstruct(U, n)

        _, uL, pL, _, aL = primitives(UL, A, n)  # primitives at x_{i-1/2} left state
        _, uR, pR, _, aR = primitives(UR, A, n)  # primitives at x_{i-1/2} right state

        gamma_exp = (k - 1) / (2 * k)
        power_exp = 2 * k / (k - 1)
        base_num = aL + aR - 0.5 * (k - 1) * (uR - uL)
        base_den = aL / (pL ** gamma_exp) + aR / (pR ** gamma_exp)
        base = base_num / base_den
        pstarr = np.power(base, power_exp)  # Toro's p* estimate at x_{i-1/2}

        qL = [1 if pstarr[i] <= pL[i] else np.sqrt(1+(k+1)/(2*k)*(pstarr[i]/pL[i]-1)) for i in range(len(pL))]
        qR = [1 if pstarr[i] <= pR[i] else np.sqrt(1+(k+1)/(2*k)*(pstarr[i]/pR[i]-1)) for i in range(len(pR))]

        SL = uL - aL*qL
        SR = uR + aR*qR
        
        if case == 'dt':
            return np.max(np.abs([SL, SR]))
        elif case == 'flux':
            return SL, SR

    def Euler_flux(U, A, n):
        rho, u, p, E, a = primitives(U, A, n)
        return np.vstack((rho*u, rho*u**2 + p, u*(rho*E + p)))

    def HLLC_flux(U, A, n):
        UL, UR, eigenvals = weno5_reconstruct(U, n)
        rhoL, uL, pL, EL, aL = primitives(UL, A, n)  # primitives at left state
        rhoR, uR, pR, ER, aR = primitives(UR, A, n)  # primitives at right state

        fL = Euler_flux(UL, A, n)
        fR = Euler_flux(UR, A, n)

        if wave_speed_method == 'Dava':
            SL = np.minimum(np.min(eigenvals, axis=0), 0)  # Davidson's HLL wave speed estimates
            SR = np.maximum(np.max(eigenvals, axis=0), 0)  # Davidson's HLL wave speed estimates
        elif wave_speed_method == 'Toro':
            SL, SR = max_wave_speed_Toro(U, A, n, case='flux')  # Toro's max wave speeds at x_{i-1/2} and x_{i+1/2}

        Sstar = (pR - pL + rhoL*uL*(SL-uL) - rhoR*uR*(SR-uR))/(rhoL*(SL-uL) - rhoR*(SR-uR))

        UstarL = rhoL * (SL - uL) / (SL - Sstar) * np.array([np.ones(SL.shape), Sstar, EL + (Sstar - uL)*(Sstar + pL/(rhoL*(SL-uL)))])
        UstarR = rhoR * (SR - uR) / (SR - Sstar) * np.array([np.ones(SL.shape), Sstar, ER + (Sstar - uR)*(Sstar + pR/(rhoR*(SR-uR)))])
        fstarL = fL + SL * (UstarL - UL)
        fstarR = fR + SR * (UstarR - UR)

        fm = np.zeros_like(UL)
        fp = np.zeros_like(UR)
        f = np.zeros_like(UL)

        for i in range(UL.shape[1]):
            if SL[i] >= 0:
                f[:, i] = fL[:, i]
            elif SL[i] < 0 and Sstar[i] >= 0:
                f[:, i] = fstarL[:, i]
            elif SR[i] > 0 and Sstar[i] <= 0:
                f[:, i] = fstarR[:, i]
            elif SR[i] <= 0:
                f[:, i] = fR[:, i]

        fm = f[:, :-1]  # flux at i-1/2
        fp = f[:, 1:]   # flux at i+1/2

        return fm, fp # flux at i-1/2 and i+1/2

    def find_dt(U, A, dx, cfl):
        if wave_speed_method == 'Dava':
            llam = max_wave_speed_Dava(U, A, n, case='dt')  # Simple
        elif wave_speed_method == 'Toro':
            llam = max_wave_speed_Toro(U, A, n, case='dt') # Toro
        return cfl * dx / llam if llam > 0 else 1e-6
    
    def SSPRK45(U, U1, U2, U3, U4, A, dt, dx, n, nx):

        # shock reflection NOT depicted accurately for now due to sboundary conditions
        U = Riemann_BC(U, n)
        fm, fp = HLLC_flux(U, A, n)
        k1 = -1/dx * (fp - fm)
        U1[:, 3:nx-3, n] = U[:, 3:nx-3, n] + 0.391752226571890*dt*k1

        U1 = Riemann_BC(U1, n)
        fm, fp = HLLC_flux(U1, A, n)
        k2 = -1/dx * (fp - fm)
        U2[:, 3:nx-3, n] = 0.444370493651235*U[:, 3:nx-3, n] + 0.555629506348765*U1[:, 3:nx-3, n] + 0.368410593050371*dt*k2

        U2 = Riemann_BC(U2, n)
        fm, fp = HLLC_flux(U2, A, n)
        k3 = -1/dx * (fp - fm)
        U3[:, 3:nx-3, n] = 0.620101851488403*U[:, 3:nx-3, n] + 0.379898148511597*U2[:, 3:nx-3, n] + 0.251891774271694*dt*k3

        U3 = Riemann_BC(U3, n)
        fm, fp = HLLC_flux(U3, A, n)
        k4 = -1/dx * (fp - fm)
        U4[:, 3:nx-3, n] = 0.178079954393132*U[:, 3:nx-3, n] + 0.821920045606868*U3[:, 3:nx-3, n] +  0.544974750228521*dt*k4

        U4 = Riemann_BC(U4, n)
        fm, fp = HLLC_flux(U4, A, n)
        k5 = -1/dx * (fp - fm)
        Unp1 = 0.517231671970585*U2[:, 3:nx-3, n] + 0.096059710526147*U3[:, 3:nx-3, n] + 0.063692468666290*dt*k4 + 0.386708617503268*U4[:, 3:nx-3, n] + 0.226007483236906*dt*k5

        return Unp1
    
    def plot(U, A, n):

        rho, u, p, E, a = primitives(U, A, n)

        os.makedirs(f"cfl{cfl}_beta{beta}_{wave_speed_method}", exist_ok=True)

        fig, ax = plt.subplots(figsize=(height, width), dpi=dpi)
        ax.plot(rho, linewidth=4, color=line_color)
        ax.set_ylabel("Densitate rho", fontsize=label_size)
        ax.set_xlabel("x [m]", fontsize=label_size)
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.grid(True)
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)
        ax.set_aspect(ar)
        plt.savefig(f"cfl{cfl}_beta{beta}_{wave_speed_method}/rho_end.png")
        # plt.clf()

        fig, ax = plt.subplots(figsize=(height, width), dpi=dpi)
        ax.plot(p, linewidth=4, color=line_color)
        ax.set_ylabel("Presiune p [Pa]", fontsize=label_size)
        ax.set_xlabel("x [m]", fontsize=label_size)
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.grid(True)
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)
        ax.set_aspect(ar)
        plt.savefig(f"cfl{cfl}_beta{beta}_{wave_speed_method}/p_end.png")
        # plt.clf()

        fig, ax = plt.subplots(figsize=(height, width), dpi=dpi)
        ax.plot(u, linewidth=4, color=line_color)
        ax.set_ylabel("Viteza u [m/s]", fontsize=label_size)
        ax.set_xlabel("x [m]", fontsize=label_size)
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.grid(True)
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)
        ax.set_aspect(ar)
        plt.savefig(f"cfl{cfl}_beta{beta}_{wave_speed_method}/u_end.png")
        # plt.clf()
    
    # Main simulation loop

    k = 1.4
    Rgas = 287  # J/kg-K

    # State
    t = 0.0
    t_end = 6e-4
    nx = 600
    dx = 1 / (nx - 6)
    cfl = 0.9
    beta = 1  # Rusanov parameter, can be tuned for stability/dissipation
    wave_speed_method = 'Toro'  # 'Dava' or 'Toro'

    # Variables to track over time
    U = np.zeros((3, nx, 2000), dtype=np.float64)  # U[0] = rho*A, U[1] = rho*u*A, U[2] = E*A; shape (nvar, nx, nt) with ghost cells for BCs; will slice to current time step
    U1 = np.zeros((3, nx, 2000), dtype=np.float64)
    U2 = np.zeros((3, nx, 2000), dtype=np.float64)
    U3 = np.zeros((3, nx, 2000), dtype=np.float64)
    U4 = np.zeros((3, nx, 2000), dtype=np.float64)
    A = np.ones([1])  # test cross-sectional area

    # Outputs
    t_list = np.array([], dtype=np.float64)
    t_list = np.append(t_list, t)
    
    n = 0
    U = initial_Riemann(U, A, n)


    while t < t_end:
        dt = find_dt(U, A, dx, cfl)
        U[:, 3:nx-3, n+1] = SSPRK45(U, U1, U2, U3, U4, A, dt, dx, n, nx)
        U = Riemann_BC(U, n+1)
        t += dt
        n += 1

        # Store outputs
        rho, u, p, E, a = primitives(U, A, n)
        t_list = np.append(t_list, t)
        print(t)

    plot(U, A, n)


if __name__ == "__main__":
    simulate()
