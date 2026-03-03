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
            E   = U[2, :, n] / rho     # total specific energy
        else:
            rho = np.maximum(U[0, :], rho_floor)
            u   = U[1, :] / rho
            E   = U[2, :] / rho    # total specific energy
        
        p = np.maximum(rho * (k - 1.0) * (E - 0.5 * u*u), p_floor)
        a = np.sqrt(np.maximum(k * p / rho, 0.0))
        # shape is (nx,) for all primitives
        return rho, u, p, E, a
    
    def initial_Riemann(U, A, n):
        shape = U.shape

        rhoL = 1.0
        pL = 1.0e5
        uL = 0.0
        left_state = np.array([rhoL, rhoL*uL, (pL/(k-1) + 0.5*rhoL*uL*uL)])  # (rho*A, rho*u*A, E*A) left state

        rhoR = 0.125
        pR = 1.0e4
        uR = 0.0
        right_state = np.array([rhoR, rhoR*uR, (pR/(k-1) + 0.5*rhoR*uR*uR)])  # (rho*A, rho*u*A, E*A) right state
        
        UL = np.repeat(left_state, shape[1]//2, axis=0).reshape(3, shape[1]//2)
        UR = np.repeat(right_state, shape[1]//2, axis=0).reshape(3, shape[1]//2)
        # even number of points necessary!
        U[:, :shape[1]//2, n] = UL
        U[:, shape[1]//2:, n] = UR
        # print(U[:, :, n])
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

            # print("rhohat:", rhohat)
            # print("uhat:", uhat)
            # print("Hhat:", Hhat)
            # print("ahat:", ahat)

            # Amatt = np.array([[0, 1, 0],
            #             [0.5*(k-3)*uhat**2, (3-k)*uhat, k-1],
            #             [0.5*uhat*(-2*Hhat+uhat*uhat*(k-1)), Hhat - uhat*uhat*(k-1), k*uhat]])

            # R = np.array([
            #     [1.0, 1.0, 1.0],
            #     [uhat - ahat, uhat, uhat + ahat],
            #     [Hhat - uhat * ahat, 0.5 * uhat * uhat, Hhat + uhat * ahat],
            # ])
            # L = np.linalg.inv(R)

            Pmatrix = np.array([[1, rhohat/(2*ahat), rhohat/(2*ahat)],
                                [uhat, rhohat*(uhat+ahat)/(2*ahat), rhohat*(uhat-ahat)/(2*ahat)],
                                [uhat*uhat/2, rhohat*(Hhat+uhat*ahat)/(2*ahat), rhohat*(Hhat-uhat*ahat)/(2*ahat)]])

            eigenvalues = np.array([uhat, uhat + ahat, uhat - ahat])
            # Matrices containing the eigenvectors ([r1, r2, r3] as columns) and their inverses
            # Pmatrix = np.linalg.eig(Amatt)[1]
            # print(Pmatrix)
            Pmatrix_inverse = np.linalg.inv(Pmatrix)
            # print("eigenvectors:\n", Pmatrix)
            # print("eigenvectors inverted:\n", Pmatrix_inverse)
            # if np.linalg.eig(A)[0][0] !=0:
            # print("eigenvalues:\n", eigenvalues)
            # print("Eigenvectors:\n", Pmatrix)
            # print("Inverse of eigenvectors:\n", Pmatrix_inverse)
            # raise ValueError("Stop")
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
        UmL = np.zeros((nvar, nc), dtype=Ulast.dtype)
        UmR = np.zeros((nvar, nc), dtype=Ulast.dtype)
        UpL = np.zeros((nvar, nc), dtype=Ulast.dtype)
        UpR = np.zeros((nvar, nc), dtype=Ulast.dtype)
        eigenvals_m = np.zeros((nvar, nc), dtype=Ulast.dtype)  # to store eigenvalues at x_{i-1/2}
        eigenvals_p = np.zeros((nvar, nc), dtype=Ulast.dtype)  # to store eigenvalues at x_{i+1/2}

        for j, ic in enumerate(range(3, nx - 3)):
            # Conservative states around i
            u_im3 = Ulast[:, ic - 3]
            u_im2 = Ulast[:, ic - 2]
            u_im1 = Ulast[:, ic - 1]
            u_i = Ulast[:, ic]
            u_ip1 = Ulast[:, ic + 1]
            u_ip2 = Ulast[:, ic + 2]
            u_ip3 = Ulast[:, ic + 3]

            # Characteristic bases at interfaces x_{i-1/2} and x_{i+1/2}
            Lm, Rm, eigenvals_m_j = eig_lr(u_im1, u_i)
            Lp, Rp, eigenvals_p_j = eig_lr(u_i, u_ip1)

            # Project stencil points to characteristic space (minus and plus interfaces)
            wm_im3 = Lm @ u_im3
            wm_im2 = Lm @ u_im2
            wm_im1 = Lm @ u_im1
            wm_i = Lm @ u_i
            wm_ip1 = Lm @ u_ip1
            wm_ip2 = Lm @ u_ip2

            wp_im2 = Lp @ u_im2
            wp_im1 = Lp @ u_im1
            wp_i = Lp @ u_i
            wp_ip1 = Lp @ u_ip1
            wp_ip2 = Lp @ u_ip2
            wp_ip3 = Lp @ u_ip3

            w_umL = np.zeros(nvar, dtype=Ulast.dtype)
            w_umR = np.zeros(nvar, dtype=Ulast.dtype)
            w_upL = np.zeros(nvar, dtype=Ulast.dtype)
            w_upR = np.zeros(nvar, dtype=Ulast.dtype)

            for pidx in range(nvar):
                # UmL: U^- at x_{i-1/2} = left ENO3 at (i-1)+1/2
                w_umL[pidx] = eno3_left_iphalf(
                    wm_im3[pidx], wm_im2[pidx], wm_im1[pidx], wm_i[pidx], wm_ip1[pidx]
                )
                # UmR: U^+ at x_{i-1/2}
                w_umR[pidx] = eno3_right_iphalf(
                    wm_im2[pidx], wm_im1[pidx], wm_i[pidx], wm_ip1[pidx], wm_ip2[pidx]
                )
                # UpL: U^- at x_{i+1/2}
                w_upL[pidx] = eno3_left_iphalf(
                    wp_im2[pidx], wp_im1[pidx], wp_i[pidx], wp_ip1[pidx], wp_ip2[pidx]
                )
                # UpR: U^+ at x_{i+1/2}
                w_upR[pidx] = eno3_right_iphalf(
                    wp_im1[pidx], wp_i[pidx], wp_ip1[pidx], wp_ip2[pidx], wp_ip3[pidx]
                )

            # Back to conservative variables
            UmL[:, j] = Rm @ w_umL
            UmR[:, j] = Rm @ w_umR
            UpL[:, j] = Rp @ w_upL
            UpR[:, j] = Rp @ w_upR
            eigenvals_m[:, j] = eigenvals_m_j
            eigenvals_p[:, j] = eigenvals_p_j

        return UmL, UmR, UpL, UpR, eigenvals_m, eigenvals_p
    
    # Wave speed
    # Option 1 (very simple): max local characteristic speed from primitives
    def max_wave_speed(U, A, n):
        _, u, _, _, a = primitives(U, A, n)
        return np.max(np.abs(u) + a)
    
    # Option 2: use nozzle exit velocity as upper bound (conservative for stability)
    # def max_wave_speed(U, A):s
    #     _, _, _, _, a = primitives(U, A)
    #     _, _, v_exit, _ = nozzle_perf(p, At, eps, k, Rgas, T1, p_amb)
    #     return max(np.max(np.abs(a)), v_exit * 1.05)  # add 5% margin for safety

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
        # llam = max_wave_speed(U, A, n)
        beta = 0.5  # Rusanov parameter, can be tuned for stability/dissipation

        UmL, UmR, UpL, UpR, eigenvals_m, eigenvals_p = eno3_reconstruct(U, n)

        # print("UmL shape:", UmL.shape)
        AmL = Amatrices(UmL, A, n)
        AmR = Amatrices(UmR, A, n)
        ApL = Amatrices(UpL, A, n)
        ApR = Amatrices(UpR, A, n)

        # Rusanov Flux at u_i-1/2:
        # Apply matrix-vector multiplication for each spatial point
        fm = np.zeros_like(UmL)
        fp = np.zeros_like(UpL)

        for i in range(UmL.shape[1]):
            alpha_m = np.max(np.abs(eigenvals_m[:, i]))
            alpha_p = np.max(np.abs(eigenvals_p[:, i]))
            fm[:, i] = 0.5 * (AmL[:, :, i] @ UmL[:, i] + AmR[:, :, i] @ UmR[:, i]) - 0.5 * beta * alpha_m * (UmR[:, i] - UmL[:, i])
            fp[:, i] = 0.5 * (ApL[:, :, i] @ UpL[:, i] + ApR[:, :, i] @ UpR[:, i]) - 0.5 * beta * alpha_p * (UpR[:, i] - UpL[:, i])


        return fm, fp # flux at i-1/2 and i+1/2
    
    def find_dt(U, A, dx, cfl):
        llam = max_wave_speed(U, A, n)
        return cfl * dx / llam if llam > 0 else 1e-6
    
    def step_RK3(U, U1, U2, A, dt, dx, n, nx):

        U = Riemann_BC(U, n)
        U = enforce_positivity(U, n, nx)
        k1 = -1/dx * (Rusanov_flux(U, A, n)[1] - Rusanov_flux(U, A, n)[0])
        U1[:, 3:nx-3, n] = U[:, 3:nx-3, n] + dt*k1
        U1 = enforce_positivity(U1, n, nx)

        U1 = Riemann_BC(U1, n)
        k2 = -1/dx * (Rusanov_flux(U1, A, n)[1] - Rusanov_flux(U1, A, n)[0])
        U2[:, 3:nx-3, n] = 0.75*U[:, 3:nx-3, n] + 0.25*(U1[:, 3:nx-3, n] + dt*k2)
        U2 = enforce_positivity(U2, n, nx)

        U2 = Riemann_BC(U2, n)
        k3 = -1/dx * (Rusanov_flux(U2, A, n)[1] - Rusanov_flux(U2, A, n)[0])

        Unp1 = (1/3)*U[:, 3:nx-3, n] + (2/3)*(U2[:, 3:nx-3, n] + dt*k3)
        return enforce_positivity_slice(Unp1)
    
    
    # Main simulation loop

    k = 1.4
    Rgas = 287  # J/kg-K

    # State
    t = 0.0
    t_end = 6e-4
    nx = 600
    dx = 1 / (nx - 6)

    # Variables to track over time
    U = np.zeros((3, nx, 2000), dtype=np.float64)  # U[0] = rho*A, U[1] = rho*u*A, U[2] = E*A; shape (nvar, nx, nt) with ghost cells for BCs; will slice to current time step
    U1 = np.zeros((3, nx, 2000), dtype=np.float64)
    U2 = np.zeros((3, nx, 2000), dtype=np.float64)
    A = np.ones([1])  # test cross-sectional area

    # Outputs
    t_list = np.array([], dtype=np.float64)
    
    n = 0
    U = initial_Riemann(U, A, n)
    rho, u, p, E, a = primitives(U, A, n)

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(rho, label='rho')
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(p, label='p')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(u, label='u')
    plt.legend()
    plt.tight_layout()
    plt.show()

    t_list = np.append(t_list, t)

    while t < t_end:
        dt = find_dt(U, A, dx, cfl=0.5)
        U[:, 3:nx-3, n+1] = step_RK3(U, U1, U2, A, dt, dx, n, nx)
        U = Riemann_BC(U, n+1)
        t += dt
        n += 1

        # Store outputs
        rho, u, p, E, a = primitives(U, A, n)
        t_list = np.append(t_list, t)
        print(t)

    #plot density, pressure, velocity at final time step
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(rho, label='rho')
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(p, label='p')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(u, label='u')
    plt.legend()
    plt.tight_layout()
    plt.show()
        

if __name__ == "__main__":
    simulate()
