#include "input_vegaE.h"

#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const double Tp0 = 300.0;
const double T2prop = 1825.0;
const double TSurf = 900.0;
const double T1prop = 3000.0;
const double Ksolid = 0.1;
const double cpsolid = 1230.0;

const double arb = 1e-4;
const double nrb = 0.4;
const double rhosolid = 1810.0;

const double mugas = 9e-5;
const double Kgas = 0.587;
const double cpgas = 2361.0;
const double kgas = 1.1487;
const double Rgas = 2361.0 * (1.1487 - 1.0) / 1.1487;
const double hreaction = 7e6;

const double kigniter = 1.22;
const double cpgas_igniter = 1900.0;
const double Rgas_igniter = 1900.0 * (1.22 - 1.0) / 1.22;
const double mugas_igniter = 8e-5;
const double Kgas_igniter = 0.4;
const double mig = 179.0;
const double vinj = 1500.0;
const double hig = 3.5e6;

const double pamb = 101325.0;
const double p0 = 101325.0;
const double rho0 = 1.225;

const double Lcase = 8.69;
const double Rcase = 1.45;
const double Vfree = 8.58;
const double Ainitial = 1.0;
const double Pinitial = 2.0 * sqrt(1.0 * M_PI);
const double A2nozzle = M_PI * 1.76 * 1.76 / 4.0;
const double epsnozzle = 16.0;

double Cstar(double T1, double k, double Rg) {
    return sqrt(Rg * T1 / k) * pow((k + 1.0) * 0.5, (k + 1.0) / (2.0 * (k - 1.0)));
}

double nozzle_mdot_from_p(double p1, double At, double cstar) {
    return p1 * At / cstar;
}

double area_mach_function(double M, double k) {
    return (1.0 / M) * pow((1.0 + 0.5 * (k - 1.0) * M * M) / (0.5 * (k + 1.0)), (k + 1.0) / (2.0 * (k - 1.0)));
}

double solve_exit_mach(double area_ratio, double k) {
    double M = fmax(1.5, 0.5 * area_ratio + 1.0);
    for (int it = 0; it < 50; ++it) {
        double f = area_mach_function(M, k) - area_ratio;
        double dM = 1e-5 * fmax(1.0, M);
        double df = (area_mach_function(M + dM, k) - area_mach_function(M - dM, k)) / (2.0 * dM);
        double step = -f / df;
        M = fmax(1.01, M + step);
        if (fabs(step) < 1e-8) {
            break;
        }
    }
    return M;
}

double solve_area_mach(double area_ratio, double k_local, int supersonic) {
    if (area_ratio <= 1.0) {
        return 1.0;
    }

    double lo;
    double hi;
    if (supersonic) {
        lo = 1.0 + 1e-8;
        hi = fmax(2.0, 0.5 * area_ratio + 1.0);
        while (area_mach_function(hi, k_local) < area_ratio && hi < 100.0) {
            hi *= 1.5;
        }
    } else {
        lo = 1e-8;
        hi = 1.0 - 1e-8;
    }

    for (int it = 0; it < 80; ++it) {
        double mid = 0.5 * (lo + hi);
        double f_mid = area_mach_function(mid, k_local) - area_ratio;
        if (supersonic) {
            if (f_mid > 0.0) {
                hi = mid;
            } else {
                lo = mid;
            }
        } else {
            if (f_mid > 0.0) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
    }
    return 0.5 * (lo + hi);
}

void nozzle_perf(double p1, double At, double eps, double k, double Rg, double T1, double p_amb,
                 double* F, double* mdot, double* v_e, double* p2_over_p1) {
    double cstar = Cstar(T1, k, Rg);
    double md = nozzle_mdot_from_p(p1, At, cstar);
    double A2 = eps * At;
    double M2 = solve_exit_mach(eps, k);
    double T2 = T1 / (1.0 + 0.5 * (k - 1.0) * M2 * M2);
    double p2 = p1 * pow(T2 / T1, k / (k - 1.0));
    double a2 = sqrt(k * Rg * T2);
    double ve = M2 * a2;
    *F = md * ve + (p2 - p_amb) * A2;
    *mdot = md;
    *v_e = ve;
    *p2_over_p1 = p2 / p1;
}

double alpha_er_from_properties(double cp_gas, double mu_gas, double K, double cs_solid,
                                double rho_p, double T1, double Ts, double T2, double Tp) {
    cp_gas = cp_gas * 0.239 / 1000.0;
    cs_solid = cs_solid * 0.239 / 1000.0;
    double Pr = mu_gas * cp_gas * 4184.0 / K;
    if ((T2 - Tp) == 0.0) {
        return 0.0;
    }
    return 0.0288 * cp_gas * pow(mu_gas, 0.2) * pow(Pr, -2.0 / 3.0) / (rho_p * cs_solid) * ((T1 - Ts) / (T2 - Tp));
}

const double alpha_er =
    0.0288 * ((2361.0 * 0.239 / 1000.0) * 0.239 / 1000.0) * pow(9e-5, 0.2) *
    pow((9e-5) * ((2361.0 * 0.239 / 1000.0) * 0.239 / 1000.0) * 4184.0 / 0.587, -2.0 / 3.0) /
    (1810.0 * ((1230.0 * 0.239 / 1000.0) * 0.239 / 1000.0)) * ((3000.0 - 900.0) / (1825.0 - 300.0));

const double beta_er = 53.0;
