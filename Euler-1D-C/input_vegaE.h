#ifndef INPUT_VEGAE_H
#define INPUT_VEGAE_H

#ifdef __cplusplus
extern "C" {
#endif

double Cstar(double T1, double k, double Rgas);
double nozzle_mdot_from_p(double p1, double At, double cstar);
double area_mach_function(double M, double k);
double solve_exit_mach(double area_ratio, double k);
double solve_area_mach(double area_ratio, double k_local, int supersonic);
void nozzle_perf(double p1, double At, double eps, double k, double Rgas, double T1, double p_amb,
                 double* F, double* mdot, double* v_e, double* p2_over_p1);
double alpha_er_from_properties(double cp_gas, double mu_gas, double K, double cs_solid,
                                double rho_p, double T1, double Ts, double T2, double Tp);

extern const double Tp0;
extern const double T2prop;
extern const double TSurf;
extern const double T1prop;
extern const double Ksolid;
extern const double cpsolid;
extern const double arb;
extern const double nrb;
extern const double rhosolid;
extern const double mugas;
extern const double Kgas;
extern const double cpgas;
extern const double kgas;
extern const double Rgas;
extern const double hreaction;
extern const double kigniter;
extern const double cpgas_igniter;
extern const double Rgas_igniter;
extern const double mugas_igniter;
extern const double Kgas_igniter;
extern const double mig;
extern const double vinj;
extern const double hig;
extern const double pamb;
extern const double p0;
extern const double rho0;
extern const double Lcase;
extern const double Rcase;
extern const double Vfree;
extern const double Ainitial;
extern const double Pinitial;
extern const double A2nozzle;
extern const double epsnozzle;
extern const double alpha_er;
extern const double beta_er;

#ifdef __cplusplus
}
#endif

#endif
