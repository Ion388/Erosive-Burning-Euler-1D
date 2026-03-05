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
    rho_floor = 1e-10
    p_floor = 1e-8

    def enforce_positivity(Uarr, n_idt, nx_local):
        rho = np.maximum(Uarr[0, :, n_idt], rho_floor)
        mom = Uarr[1, :, n_idt]
        Et = Uarr[2, :, n_idt]

        kinetic = 0.5 * mom * mom / rho
        Et_min = kinetic + p_floor / (k - 1.0)

        Uarr[0, :, n_idt] = rho
        Uarr[2, :, n_idt] = np.maximum(Et, Et_min)
        return Uarr

    def enforce_positivity_slice(U_slice):
        rho = np.maximum(U_slice[0, :], rho_floor)
        mom = U_slice[1, :]
        Et = U_slice[2, :]


        kinetic = 0.5 * mom * mom / rho
        Et_min = kinetic + p_floor / (k - 1.0)

        U_slice[0, :] = rho
        U_slice[2, :] = np.maximum(Et, Et_min)
        return U_slice
    
    def primitives(U, A, n):
        """Convert conserved variables U to primitive variables (rho, u, p, E, a).
        U = [rho*A, rho*u*A, rho*E*A], where E is total energy per volume. A is cross-sectional area.
        """
        A_place = 1
        if U.ndim == 3:
            rho = np.maximum(U[0, :, n], rho_floor)
            u   = U[1, :, n] / rho
            E   = U[2, :, n] / rho     # total specific energy (per unit mass)
        else:
            rho = np.maximum(U[0, :], rho_floor)
            u   = U[1, :] / rho
            E   = U[2, :] / rho    # total specific energy (per unit mass)
        
        p = np.maximum(rho * (k - 1.0) * (E - 0.5 * u*u), p_floor)
        a = np.sqrt(np.maximum(k * p / rho, 0.0))
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

    def eno3_reconstruct(U, n):
        """
        Characteristic ENO3 reconstruction for 1D Euler variables.

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
            raise ValueError("Need at least 7 points (including ghost cells) for ENO3.")
        if nvar != 3:
            raise ValueError("Characteristic ENO3 here is implemented for 1D Euler with 3 conserved variables.")

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

            b0 = abs(w_im2 - 2.0 * w_im1 + w_i)
            b1 = abs(w_im1 - 2.0 * w_i + w_ip1)
            b2 = abs(w_i - 2.0 * w_ip1 + w_ip2)
            s = int(np.argmin([b0, b1, b2]))
            return [p0, p1, p2][s]

        def eno3_right_iphalf(w_im1, w_i, w_ip1, w_ip2, w_ip3):
            p0 = -(1.0 / 6.0) * w_im1 + (5.0 / 6.0) * w_i + (1.0 / 3.0) * w_ip1
            p1 = (1.0 / 3.0) * w_i + (5.0 / 6.0) * w_ip1 - (1.0 / 6.0) * w_ip2
            p2 = (11.0 / 6.0) * w_ip1 - (7.0 / 6.0) * w_ip2 + (1.0 / 3.0) * w_ip3

            b0 = abs(w_im1 - 2.0 * w_i + w_ip1)
            b1 = abs(w_i - 2.0 * w_ip1 + w_ip2)
            b2 = abs(w_ip1 - 2.0 * w_ip2 + w_ip3)
            s = int(np.argmin([b0, b1, b2]))
            return [p0, p1, p2][s]

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
                # UmR: U^+ at x_{i-1/2}
                w_uR[pidx] = eno3_right_iphalf(
                    wm_im2[pidx], wm_im1[pidx], wm_i[pidx], wm_ip1[pidx], wm_ip2[pidx]
                )


            UL[:, j] = R @ w_uL
            UR[:, j] = R @ w_uR 
            eigenvals[:, j] = eigenvals_j

        return UL, UR, eigenvals
    
    # Wave speed
    # Option 1 (very simple): max local characteristic speed from primitives
    def max_wave_speed(U, A, n, case):
        UL, UR, eigenvalues = eno3_reconstruct(U, n)
        # _, uL, _, _, aL = primitives(UL, A, n)
        # _, uR, _, _, aR = primitives(UR, A, n)
        # alpha = [np.abs(uL) + aL, np.abs(uR) + aR]
        alpha = np.abs(eigenvalues)  # max eigenvalue magnitude at each interface

        if case == 'dt':
            return np.max(alpha)
        elif case == 'flux':
            return np.max(alpha, axis=0)

    def max_wave_speed_Toro(U, A, n, case):
        UL, UR, eigenvals = eno3_reconstruct(U, n)

        _, uL, pL, _, aL = primitives(UL, A, n)  # primitives at x_{i-1/2} left state
        _, uR, pR, _, aR = primitives(UR, A, n)  # primitives at x_{i-1/2} right state

        if np.any(pL <= 0) or np.any(pR <= 0):
            raise ValueError("Non-positive pressure before Toro p* estimate in max_wave_speed_Toro.")

        gamma_exp = (k - 1) / (2 * k)
        power_exp = 2 * k / (k - 1)

        base_num = aL + aR - 0.5 * (k - 1) * (uR - uL)
        base_den = aL / (pL ** gamma_exp) + aR / (pR ** gamma_exp)

        if np.any(base_den <= 0):
            raise ValueError("Non-positive denominator in Toro p* estimate in max_wave_speed_Toro.")

        base = base_num / base_den

        if np.any(~np.isfinite(base)):
            raise ValueError("Non-finite Toro p* base in max_wave_speed_Toro.")
        if np.any(base <= 0):
            raise ValueError(
                f"Non-positive Toro p* base in max_wave_speed_Toro: min(base)={np.min(base)}"
            )

        pstarr = np.power(base, power_exp)  # Toro's p* estimate at x_{i-1/2}

        qL = [1 if pstarr[i] <= pL[i] else np.sqrt(1+(k+1)/(2*k)*(pstarr[i]/pL[i]-1)) for i in range(len(pL))]
        qR = [1 if pstarr[i] <= pR[i] else np.sqrt(1+(k+1)/(2*k)*(pstarr[i]/pR[i]-1)) for i in range(len(pR))]

        SL = uL - aL*qL
        SR = uR + aR*qR
        
        if case == 'dt':
            return np.max(np.abs([SL, SR]))
        elif case == 'flux':
            return np.max(np.abs([SL, SR]), axis=0)
    

    def Amatrices(U, A, n):
        rho, u, p, E, a = primitives(U, A, n)
        U_shape = U.shape
        Amatrices = np.zeros((3, 3, U_shape[1]))  # (nvar, nvar, nx)
        for i in range(U_shape[1]):
            Amatrices[:, :, i] = np.array([[0, 1, 0],
                        [0.5*(k-3)*u[i]**2, (3-k)*u[i], k-1],
                        [u[i]*(0.5*(k-1)*u[i]**2 - k*E[i]), E[i]*k - 3/2*(k-1)*u[i]**2, k*u[i]]])
        return Amatrices

    def Rusanov_flux(U, A, n):

        UL, UR, eigenvals = eno3_reconstruct(U, n)

        # print("UL:", UL)
        # print("UR:", UR)

        # for i in range(UL.shape[1]-1):
        #     if np.all(UL[:, i+1] == UR[:, i]):
        #         print(f"Warning: UL and UR are identical at index {i}, which may cause zero wave speed and excessive dissipation in Rusanov flux.")

        if wave_speed_method == 'simple':
            alpha = max_wave_speed(U, A, n, case='flux')  # Simple max wave speeds at x_{i-1/2} and x_{i+1/2}

        elif wave_speed_method == 'Toro':
            alpha = max_wave_speed_Toro(U, A, n, case='flux')  # Toro's max wave speeds at x_{i-1/2} and x_{i+1/2}
            
        AL = Amatrices(UL, A, n)
        AR = Amatrices(UR, A, n)

        f = np.zeros_like(UL)

        for i in range(UL.shape[1]):
            f[:, i] = 0.5 * (AL[:, :, i] @ UL[:, i] + AR[:, :, i] @ UR[:, i]) - 0.5 * beta * alpha[i] * (UR[:, i] - UL[:, i])

        fm = f[:, :-1]  # flux at i-1/2
        fp = f[:, 1:]   # flux at i+1/2

        return fm, fp # flux at i-1/2 and i+1/2
    
    def HLL_flux(U, A, n):
        UL, UR, eigenvals = eno3_reconstruct(U, n)

        if wave_speed_method == 'simple':
            alpha = max_wave_speed(U, A, n, case='flux')  # Simple max wave speeds at x_{i-1/2} and x_{i+1/2}

        elif wave_speed_method == 'Toro':
            alpha = max_wave_speed_Toro(U, A, n, case='flux')  # Toro's max wave speeds at x_{i-1/2} and x_{i+1/2}
            
        AL = Amatrices(UL, A, n)
        AR = Amatrices(UR, A, n)

        fL = np.zeros_like(UL)
        fR = np.zeros_like(UR)

        for i in range(UL.shape[1]):
            fL[:, i] = AL[:, :, i] @ UL[:, i]
            fR[:, i] = AR[:, :, i] @ UR[:, i]

        SL = np.minimum(np.min(eigenvals, axis=0), 0)  # Davidson's HLL wave speed estimates
        SR = np.maximum(np.max(eigenvals, axis=0), 0)  # Davidson's HLL wave speed estimates

        fm = np.zeros_like(UL)
        fp = np.zeros_like(UR)
        f = np.zeros_like(UL)

        for i in range(UL.shape[1]):
            if SL[i] >= 0:
                f[:, i] = fL[:, i]
                # fp[:, i] = fL[:, i]
            elif SR[i] <= 0:
                f[:, i] = fR[:, i]
                # fp[:, i] = fR[:, i]
            else:
                f[:, i] = (SR[i]*fL[:, i] - SL[i]*fR[:, i] + SL[i]*SR[i]*(UR[:, i] - UL[:, i])) / (SR[i] - SL[i])
                # fp[:, i] = fm[:, i]

        fm = f[:, :-1]  # flux at i-1/2
        fp = f[:, 1:]   # flux at i+1/2

        # print(fm.shape)
        return fm, fp # flux at i-1/2 and i+1/2

    def HLLC_flux(U, A, n):
        UL, UR, eigenvals = eno3_reconstruct(U, n)
    
    def find_dt(U, A, dx, cfl):
        if wave_speed_method == 'simple':
            llam = max_wave_speed(U, A, n, case='dt')  # Simple
        elif wave_speed_method == 'Toro':
            llam = max_wave_speed_Toro(U, A, n, case='dt') # Toro
        return cfl * dx / llam if llam > 0 else 1e-6
    
    def step_RK3(U, U1, U2, A, dt, dx, n, nx):

        # shock reflection NOT depicted accurately for now
        U = Riemann_BC(U, n)
        U = enforce_positivity(U, n, nx)
        fm, fp = HLL_flux(U, A, n)
        k1 = -1/dx * (fp - fm)
        U1[:, 3:nx-3, n] = U[:, 3:nx-3, n] + dt*k1

        U1 = Riemann_BC(U1, n)
        U1 = enforce_positivity(U1, n, nx)
        fm, fp = HLL_flux(U1, A, n)
        k2 = -1/dx * (fp - fm)
        U2[:, 3:nx-3, n] = 0.75*U[:, 3:nx-3, n] + 0.25*(U1[:, 3:nx-3, n] + dt*k2)

        U2 = Riemann_BC(U2, n)
        U2 = enforce_positivity(U2, n, nx)
        fm, fp = HLL_flux(U2, A, n)
        k3 = -1/dx * (fp - fm)
        Unp1 = (1/3)*U[:, 3:nx-3, n] + (2/3)*(U2[:, 3:nx-3, n] + dt*k3)

        return enforce_positivity_slice(Unp1)
    
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
    wave_speed_method = 'simple'  # 'simple' or 'Toro'

    # Variables to track over time
    U = np.zeros((3, nx, 2000), dtype=np.float64)  # U[0] = rho*A, U[1] = rho*u*A, U[2] = E*A; shape (nvar, nx, nt) with ghost cells for BCs; will slice to current time step
    U1 = np.zeros((3, nx, 2000), dtype=np.float64)
    U2 = np.zeros((3, nx, 2000), dtype=np.float64)
    A = np.ones([1])  # test cross-sectional area

    # Outputs
    t_list = np.array([], dtype=np.float64)
    t_list = np.append(t_list, t)
    
    n = 0
    U = initial_Riemann(U, A, n)


    while t < t_end:
        dt = find_dt(U, A, dx, cfl)
        U[:, 3:nx-3, n+1] = step_RK3(U, U1, U2, A, dt, dx, n, nx)
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
