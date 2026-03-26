import numpy as np

def Cstar(T1, k, Rgas):
    return np.sqrt(Rgas * T1 / k) * ((k + 1.0) * 0.5) ** ((k + 1.0) / (2.0 * (k - 1.0)))

def nozzle_mdot_from_p(p1, At, cstar):
    return p1 * At / cstar

def area_mach_function(M, k):
    return (1.0 / M) * (((1.0 + 0.5 * (k - 1.0) * M * M)/(0.5 * (k + 1))) ** ((k + 1.0) / (2.0 * (k - 1.0))))

def solve_exit_mach(area_ratio, k):
    # Simple Newton-Raphson with clamp; start at M=2.5
    M = max(1.5, 0.5 * area_ratio + 1.0)
    # M = 3
    for _ in range(50):
        f = area_mach_function(M, k) - area_ratio
        # Numerical derivative
        dM = 1e-5 * max(1.0, M)
        df = (area_mach_function(M + dM, k) - area_mach_function(M - dM, k)) / (2.0 * dM)
        step = -f / df
        M = max(1.01, M + step)
        if abs(step) < 1e-8:
            break
    return M

def solve_area_mach(area_ratio, k_local, supersonic):
        if area_ratio <= 1.0:
            return 1.0

        if supersonic:
            lo = 1.0 + 1e-8
            hi = max(2.0, 0.5 * area_ratio + 1.0)
            while area_mach_function(hi, k_local) < area_ratio and hi < 100.0:
                hi *= 1.5
        else:
            lo = 1e-8
            hi = 1.0 - 1e-8

        for _ in range(80):
            mid = 0.5 * (lo + hi)
            f_mid = area_mach_function(mid, k_local) - area_ratio
            if supersonic:
                if f_mid > 0.0:
                    hi = mid
                else:
                    lo = mid
            else:
                if f_mid > 0.0:
                    lo = mid
                else:
                    hi = mid
        # print(lo, hi)
        return 0.5 * (lo + hi)


def nozzle_perf(p1, At, eps, k, Rgas, T1, p_amb):
    cstar = Cstar(T1, k, Rgas)
    mdot = nozzle_mdot_from_p(p1, At, cstar)
    A2 = eps * At
    M2 = solve_exit_mach(eps, k)
    T2 = T1 / (1.0 + 0.5 * (k - 1.0) * M2 * M2)
    p2 = p1 * (T2 / T1) ** (k / (k - 1.0))
    a_2 = np.sqrt(k * Rgas * T2)
    v_e = M2 * a_2
    F = mdot * v_e + (p2 - p_amb) * A2
    return F, mdot, v_e, p2/p1


def alpha_er_from_properties(cp_gas, mu_gas, K, cs_solid, rho_p, T1, Ts, T2, Tp):
    cp_gas = cp_gas*0.239/1000
    cs_solid = cs_solid*0.239/1000
    Pr = mu_gas * cp_gas * 4184 / K
    if (T2 - Tp) == 0:
        return 0.0
    return 0.0288 * cp_gas * (mu_gas ** 0.2) * (Pr ** (-2.0 / 3.0)) / (rho_p * cs_solid) * ((T1 - Ts) / (T2 - Tp))

# Erosive burning parameters
Tp0 = 300.0             # K, initial propellant temperature
T2prop = 1825.0         # K, propellant surface temperature at ignition
TSurf = 900.0           # K, ignition threshold at surface
T1prop = 3000.0         # K, combustion gas temperature for erosive burning calculation
# Ksolid = 0.2            # W/m-K, solid propellant thermal conductivity
Ksolid = 0.1            # W/m-K, solid propellant thermal conductivity
cpsolid = 1230.0        # J/kg-K, solid propellant specific heat

arb = 1e-4
nrb = 0.4
rhosolid = 1810.0       # kg/m^3

mugas = 9e-5            # kg/m-s
Kgas = 0.587            # W/m-K
cpgas = 2361            # J/kg-K, gas specific heat
kgas = 1.1487
Rgas = cpgas * (kgas-1) / kgas # J/kg-K
hreaction = 7e6


# Igniter parameters
kigniter = 1.22            # gamma for igniter gas, for ignition thermal state calculation 
cpgas_igniter = 1900       # J/kg-K, for ignition thermal state calculation
Rgas_igniter = cpgas_igniter * (kigniter-1) / kigniter
mugas_igniter = 8e-5         # kg/m-s, for ignition thermal state calculation    
Kgas_igniter = 0.4       # W/m-K, for ignition thermal state calculation
mig = 179.0                  # kg/s for 0.35s
vinj = 1500.0                   # m/s
hig = 3.5e6                 # J/kg

# Start values
pamb = 101325 # Pa
p0 = 101325 # Pa
rho0 = 1.225 # kg/m^3

# Nozzle and case geometry
Lcase = 8.69  # m
Rcase = 1.45 # m
Vfree = 8.58 # m^3
Ainitial = 1 # m^2
Pinitial = 2*np.sqrt(Ainitial*np.pi)
A2nozzle = np.pi*1.76*1.76/4 # m^2
epsnozzle = 16.0

# Case Geometry
# Lcase = 8.69  # m
# Rcase = 1.45 # m
# Vfree = 8.58 # m^3
# Ainitial = 1 # m^2

# # Nozzle
# A2nozzle = np.pi*1.76*1.76/4 # m^2
# epsnozzle = 16.0
# pamb = 101325 # Pa
# p0 = 101325 # Pa
# rho0 = 1.225 # kg/m^3

# Propellant
# rhoprop = 1810.0 # kg/m^3
# arb = 3e-4
# nrb = 0.4
# hreaction = 3.5e6  # reaction heat release for source term example

# Combustion gas
# k_gas = 1.1487
# # Rgas = 274.7249 # J/kg-K
# T1 = 3000.0 # K
# cpgas = 2361 # J/kg-K
# Rgas = cpgas * (k_gas-1) / k_gas # J/kg-K

# Igniter properties
# mig = 179.0 # kg/s for 0.35s
# vinj = 1500.0 # m/s
# hig = 3.5e6 # J/kg

# Erosive burning properties
# cpgas=2361*0.239/1000,    # kJ/kg-K to kcal/kg-K
# mu_gas=9e-5, # kg/m-s
# K=0.587,  # W/m-K+
# cs_solid=1230.0*0.239/1000,  # kJ/kg-K to kcal/kg-K
# rho_p=rhoprop, # kg/m^3

# TS=1060.0, # K
# T2=1825.0, # K
# Tp0=300.0, # K



# Erosive (Lenoir–Robillard)
alpha_er = alpha_er_from_properties(
    cp_gas=cpgas*0.239/1000,    # kJ/kg-K to kcal/kg-K
    mu_gas=mugas, # kg/m-s
    K=Kgas,  # W/m-K+
    cs_solid=cpsolid*0.239/1000,  # kJ/kg-K to kcal/kg-K
    rho_p=rhosolid, # kg/m^3
    T1=T1prop, # K
    Ts=TSurf, # K
    T2=1825.0, # K
    Tp=300.0, # K
)
beta_er = 53.0
# print(alpha_er)
