import numpy as np

# U = np.zeros((3, 100, 10))  # (nvar, nx, nt)
# k = 1.4
# A = np.ones([100])  # test cross-sectional area
# Ulast = np.ones_like(U)  # Initialize Ulast with the same shape as U
# U_last = Ulast[:, :, -1]  # Get the last time step from Ulast
# shape = U.shape
# left_state = np.array([1.0, 1.0, 0.0])  # (rho, p, u) left state
# right_state = np.array([0.125, 0.1, 0.0])  # (rho, p, u) right state
# UL = np.repeat(left_state, shape[1], axis=0).reshape(3, shape[1])
# UR = np.repeat(right_state, shape[1], axis=0).reshape(3, shape[1])
# U[:, :, -1] = U_last  # update U with new BCs for current time step
# print(U)

# def primitives(U, A):
#     """Convert conserved variables U to primitive variables (rho, u, p, e, a).
#     U = [rho*A, rho*u*A, e*A], where e is total energy per volume. A is cross-sectional area.
#     """
#     rho = U[0, :, -1] / A
#     u   = U[1, :, -1] / U[0, :, -1]
#     e   = U[2, :, -1] / A     # energy per unit volume

#     p = (k - 1.0) * (e - 0.5 * rho * u**2)

#     a = np.sqrt(k * p / rho)
#     # shape is (nx,) for all primitives
#     return rho, u, p, e, a

# rho, u, p, e, a = primitives(U, A)
# Amatrices = np.zeros((3, 3, 100))  # (nvar, nvar, nx)
# for i in range(100):
#     Amatrices[:, :, i] = np.array([[0, 1, 0],
#                 [0.5*(k-3)*u[i]**2, (3-k)*u[i], k-1],
#                 [u[i]*(0.5*(k-1)*u[i]**2 - k*e[i]/rho[i]), e[i]/rho[i]*k - 3/2*(k-1)*u[i]**2, k*u[i]]])
# print(Amatrices.shape)

# div1 = [1, 2, 3]
# div2 = [4, 5, 6]

# index = np.where(np.array(div1) < np.array(div2))[0]
# print(index)

# for j, ic in enumerate(range(4, 10)):
#     print(f"j={j}, ic={ic}")

np.random.seed(42)
eigenvals_m = np.random.rand(3, 10)  # (nvar, nx)
eigenvals_p = np.random.rand(3, 10)  # (nvar, nx)
print(eigenvals_m)
print(eigenvals_p)
for i in range(eigenvals_m.shape[1]):
    alpha_m = np.max(np.abs([eigenvals_m, eigenvals_p]), axis=0) #Toro

print(alpha_m)