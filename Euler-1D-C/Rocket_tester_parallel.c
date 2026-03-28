#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
#include <direct.h>
#define MKDIR(path) _mkdir(path)
#else
#include <sys/stat.h>
#define MKDIR(path) mkdir(path, 0777)
#endif

#include "input_vegaE.h"
#include "riemann_test_cases.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define NVAR 3

#define IDX(v, i, nx) ((v) * (nx) + (i))
#define MAXD(a, b) ((a) > (b) ? (a) : (b))
#define MIND(a, b) ((a) < (b) ? (a) : (b))

typedef struct {
    int nx;
    int m;
    int n_updated;
    int n_real;

    double area_floor;
    double rho_floor;
    double p_floor;
    double a2_floor;

    double* U1;
    double* U2;
    double* U3;
    double* U4;

    double* UL;
    double* UR;
    double* khat;

    double* rho;
    double* u;
    double* p;
    double* E;
    double* a;

    double* fL;
    double* fR;
    double* f;
    double* fm;
    double* fp;
    double* fm_hlle;
    double* fp_hlle;

    double* SL;
    double* SR;
    double* Sstar;

    double* S1;
    double* S2;
    double* S3;
    double* S4;
    double* S5;

    double* rb1;
    double* rb2;
    double* rb3;
    double* rb4;
    double* rb5;

    double* k1;
    double* k2;
    double* k3;
    double* k4;
    double* k5;

    int* troubled;
    double* theta;

    double* tmp_nx;
    double* tmp_nx2;
    double* tmp_nx3;
} SolverWorkspace;

typedef struct {
    int n_inner;
    double* Ts;
    unsigned char* ignited;

    double* q_hist;
    double* t_hist;
    int hist_len;
    int hist_cap;
    double time;
} ThermalState;

typedef struct {
    int nx;
    double* t;
    double* A;
    double* P;
    int len;
    int cap;
} GeometryHistory;

static double regularize_denom(double x, double eps) {
    if (fabs(x) < eps) {
        return (x >= 0.0) ? eps : -eps;
    }
    return x;
}

static double gas_constant_from_k(double k_local) {
    return (fabs(k_local - kgas) <= fabs(k_local - kigniter)) ? Rgas : Rgas_igniter;
}

static int ensure_geometry_history_capacity(GeometryHistory* gh, int needed_len) {
    if (needed_len <= gh->cap) {
        return 1;
    }
    int new_cap = gh->cap > 0 ? gh->cap : 128;
    while (new_cap < needed_len) {
        new_cap *= 2;
    }

    double* new_t = (double*)realloc(gh->t, (size_t)new_cap * sizeof(double));
    double* new_A = (double*)realloc(gh->A, (size_t)new_cap * (size_t)gh->nx * sizeof(double));
    double* new_P = (double*)realloc(gh->P, (size_t)new_cap * (size_t)gh->nx * sizeof(double));
    if (!new_t || !new_A || !new_P) {
        free(new_t);
        free(new_A);
        free(new_P);
        return 0;
    }
    gh->t = new_t;
    gh->A = new_A;
    gh->P = new_P;
    gh->cap = new_cap;
    return 1;
}

static int append_geometry_history(GeometryHistory* gh, double t, const double* A, const double* P) {
    if (!ensure_geometry_history_capacity(gh, gh->len + 1)) {
        return 0;
    }
    gh->t[gh->len] = t;
    memcpy(gh->A + (size_t)gh->len * (size_t)gh->nx, A, (size_t)gh->nx * sizeof(double));
    memcpy(gh->P + (size_t)gh->len * (size_t)gh->nx, P, (size_t)gh->nx * sizeof(double));
    gh->len += 1;
    return 1;
}

static void free_geometry_history(GeometryHistory* gh) {
    free(gh->t);
    free(gh->A);
    free(gh->P);
    memset(gh, 0, sizeof(*gh));
}

static int ensure_thermal_history_capacity(ThermalState* ts, int needed_len) {
    if (needed_len <= ts->hist_cap) {
        return 1;
    }
    int new_cap = ts->hist_cap > 0 ? ts->hist_cap : 128;
    while (new_cap < needed_len) {
        new_cap *= 2;
    }

    double* new_t_hist = (double*)realloc(ts->t_hist, (size_t)(new_cap + 1) * sizeof(double));
    double* new_q_hist = (double*)realloc(ts->q_hist, (size_t)new_cap * (size_t)ts->n_inner * sizeof(double));
    if (!new_t_hist || !new_q_hist) {
        free(new_t_hist);
        free(new_q_hist);
        return 0;
    }

    ts->t_hist = new_t_hist;
    ts->q_hist = new_q_hist;
    ts->hist_cap = new_cap;
    return 1;
}

static void free_thermal_state(ThermalState* ts) {
    free(ts->Ts);
    free(ts->ignited);
    free(ts->q_hist);
    free(ts->t_hist);
    memset(ts, 0, sizeof(*ts));
}

static int init_thermal_state(ThermalState* ts, int nx) {
    memset(ts, 0, sizeof(*ts));
    ts->n_inner = nx - 7;
    ts->Ts = (double*)malloc((size_t)ts->n_inner * sizeof(double));
    ts->ignited = (unsigned char*)calloc((size_t)ts->n_inner, sizeof(unsigned char));
    ts->t_hist = (double*)malloc(sizeof(double));
    if (!ts->Ts || !ts->ignited || !ts->t_hist) {
        free_thermal_state(ts);
        return 0;
    }
    for (int i = 0; i < ts->n_inner; ++i) {
        ts->Ts[i] = Tp0;
    }
    ts->t_hist[0] = 0.0;
    ts->time = 0.0;
    ts->hist_len = 0;
    ts->hist_cap = 0;
    return 1;
}

static int alloc_workspace(SolverWorkspace* ws, int nx) {
    memset(ws, 0, sizeof(*ws));
    ws->nx = nx;
    ws->m = nx - 5;
    ws->n_updated = nx - 6;
    ws->n_real = nx - 7;

    ws->area_floor = 1e-12;
    ws->rho_floor = MAXD(1e-9, 1e-8 * rho0);
    ws->p_floor = MAXD(1.0, 1e-8 * p0);
    ws->a2_floor = 1e-10;

    size_t nx3 = (size_t)NVAR * (size_t)nx;
    size_t m3 = (size_t)NVAR * (size_t)ws->m;
    size_t fm3 = (size_t)NVAR * (size_t)(ws->m - 1);
    size_t up3 = (size_t)NVAR * (size_t)ws->n_updated;

    ws->U1 = (double*)calloc(nx3, sizeof(double));
    ws->U2 = (double*)calloc(nx3, sizeof(double));
    ws->U3 = (double*)calloc(nx3, sizeof(double));
    ws->U4 = (double*)calloc(nx3, sizeof(double));

    ws->UL = (double*)calloc(m3, sizeof(double));
    ws->UR = (double*)calloc(m3, sizeof(double));
    ws->khat = (double*)calloc((size_t)ws->m, sizeof(double));

    ws->rho = (double*)calloc((size_t)nx, sizeof(double));
    ws->u = (double*)calloc((size_t)nx, sizeof(double));
    ws->p = (double*)calloc((size_t)nx, sizeof(double));
    ws->E = (double*)calloc((size_t)nx, sizeof(double));
    ws->a = (double*)calloc((size_t)nx, sizeof(double));

    ws->fL = (double*)calloc(m3, sizeof(double));
    ws->fR = (double*)calloc(m3, sizeof(double));
    ws->f = (double*)calloc(m3, sizeof(double));
    ws->fm = (double*)calloc(fm3, sizeof(double));
    ws->fp = (double*)calloc(fm3, sizeof(double));
    ws->fm_hlle = (double*)calloc(fm3, sizeof(double));
    ws->fp_hlle = (double*)calloc(fm3, sizeof(double));

    ws->SL = (double*)calloc((size_t)ws->m, sizeof(double));
    ws->SR = (double*)calloc((size_t)ws->m, sizeof(double));
    ws->Sstar = (double*)calloc((size_t)ws->m, sizeof(double));

    ws->S1 = (double*)calloc(up3, sizeof(double));
    ws->S2 = (double*)calloc(up3, sizeof(double));
    ws->S3 = (double*)calloc(up3, sizeof(double));
    ws->S4 = (double*)calloc(up3, sizeof(double));
    ws->S5 = (double*)calloc(up3, sizeof(double));

    ws->rb1 = (double*)calloc((size_t)ws->n_real, sizeof(double));
    ws->rb2 = (double*)calloc((size_t)ws->n_real, sizeof(double));
    ws->rb3 = (double*)calloc((size_t)ws->n_real, sizeof(double));
    ws->rb4 = (double*)calloc((size_t)ws->n_real, sizeof(double));
    ws->rb5 = (double*)calloc((size_t)ws->n_real, sizeof(double));

    ws->k1 = (double*)calloc(up3, sizeof(double));
    ws->k2 = (double*)calloc(up3, sizeof(double));
    ws->k3 = (double*)calloc(up3, sizeof(double));
    ws->k4 = (double*)calloc(up3, sizeof(double));
    ws->k5 = (double*)calloc(up3, sizeof(double));

    ws->troubled = (int*)calloc((size_t)ws->m, sizeof(int));
    ws->theta = (double*)calloc((size_t)ws->m, sizeof(double));

    ws->tmp_nx = (double*)calloc((size_t)nx, sizeof(double));
    ws->tmp_nx2 = (double*)calloc((size_t)nx, sizeof(double));
    ws->tmp_nx3 = (double*)calloc((size_t)nx, sizeof(double));

    if (!ws->U1 || !ws->U2 || !ws->U3 || !ws->U4 || !ws->UL || !ws->UR || !ws->khat || !ws->rho || !ws->u || !ws->p ||
        !ws->E || !ws->a || !ws->fL || !ws->fR || !ws->f || !ws->fm || !ws->fp || !ws->fm_hlle || !ws->fp_hlle || !ws->SL ||
        !ws->SR || !ws->Sstar || !ws->S1 || !ws->S2 || !ws->S3 || !ws->S4 || !ws->S5 || !ws->rb1 || !ws->rb2 || !ws->rb3 ||
        !ws->rb4 || !ws->rb5 || !ws->k1 || !ws->k2 || !ws->k3 || !ws->k4 || !ws->k5 || !ws->troubled || !ws->theta || !ws->tmp_nx ||
        !ws->tmp_nx2 || !ws->tmp_nx3) {
        return 0;
    }
    return 1;
}

static void free_workspace(SolverWorkspace* ws) {
    free(ws->U1);
    free(ws->U2);
    free(ws->U3);
    free(ws->U4);
    free(ws->UL);
    free(ws->UR);
    free(ws->khat);
    free(ws->rho);
    free(ws->u);
    free(ws->p);
    free(ws->E);
    free(ws->a);
    free(ws->fL);
    free(ws->fR);
    free(ws->f);
    free(ws->fm);
    free(ws->fp);
    free(ws->fm_hlle);
    free(ws->fp_hlle);
    free(ws->SL);
    free(ws->SR);
    free(ws->Sstar);
    free(ws->S1);
    free(ws->S2);
    free(ws->S3);
    free(ws->S4);
    free(ws->S5);
    free(ws->rb1);
    free(ws->rb2);
    free(ws->rb3);
    free(ws->rb4);
    free(ws->rb5);
    free(ws->k1);
    free(ws->k2);
    free(ws->k3);
    free(ws->k4);
    free(ws->k5);
    free(ws->troubled);
    free(ws->theta);
    free(ws->tmp_nx);
    free(ws->tmp_nx2);
    free(ws->tmp_nx3);
    memset(ws, 0, sizeof(*ws));
}

static void fill_geometry_ghosts(double* A, double* P, int nx) {
    A[0] = A[3];
    A[1] = A[3];
    A[2] = A[3];

    A[nx - 4] = A[nx - 5];
    A[nx - 3] = A[nx - 5];
    A[nx - 2] = A[nx - 5];
    A[nx - 1] = A[nx - 5];

    P[0] = P[3];
    P[1] = P[3];
    P[2] = P[3];

    P[nx - 4] = P[nx - 5];
    P[nx - 3] = P[nx - 5];
    P[nx - 2] = P[nx - 5];
    P[nx - 1] = P[nx - 5];
}

static void enforce_positive_state(double* U, const double* A, const double* k, const SolverWorkspace* ws) {
    const int nx = ws->nx;
    const double area_floor = ws->area_floor;
    const double rho_floor = ws->rho_floor;
    const double p_floor = ws->p_floor;

    double Ause, rho, rho_Ause, u, kminus1, E, p, E_final;
    int i;

    // #pragma omp parallel for private(i, Ause, rho, rho_Ause, u, kminus1, E, p, E_final) shared(U, A, k, ws) schedule(static, 12)
    for (i = 0; i < nx; ++i) {
        Ause = MAXD(A[i], area_floor);
        rho = MAXD(U[IDX(0, i, nx)] / Ause, rho_floor);
        rho_Ause = rho * Ause;
        u = U[IDX(1, i, nx)] / rho_Ause;

        kminus1 = MAXD(k[i] - 1.0, 1e-8);
        E = U[IDX(2, i, nx)] / rho_Ause;
        p = rho * kminus1 * (E - 0.5 * u * u);
        p = MAXD(p, p_floor);
        E_final = p / (rho * kminus1) + 0.5 * u * u;

        U[IDX(0, i, nx)] = rho * Ause;
        U[IDX(1, i, nx)] = u * rho_Ause;
        U[IDX(2, i, nx)] = rho * E_final * Ause;
    }
}

static void primitives_general(const double* U, int nU, const double* A, int nA, const double* k,
                               double* rho, double* u, double* p, double* E_final, double* a,
                               const SolverWorkspace* ws) {
    const double area_floor = ws->area_floor;
    const double rho_floor = ws->rho_floor;
    const double p_floor = ws->p_floor;
    const double a2_floor = ws->a2_floor;

    double Ause, rhoi, rho_A, ui, kminus1, Ei, pi, Efi, a2;
    int i;

    // #pragma omp parallel for private(i, Ause, rhoi, rho_A, ui, kminus1, Ei, pi, Efi, a2) shared(U, A, k, rho, u, p, E_final, a, ws) schedule(static, 12)
    for (i = 0; i < nU; ++i) {
        if (nU != nA) {
            Ause = 0.5 * (A[i + 2] + A[i + 3]);
        } else {
            Ause = A[i];
        }
        Ause = MAXD(Ause, area_floor);

        rhoi = MAXD(U[IDX(0, i, nU)] / Ause, rho_floor);
        rho_A = rhoi * Ause;
        ui = U[IDX(1, i, nU)] / rho_A;
        Ei = U[IDX(2, i, nU)] / rho_A;

        kminus1 = MAXD(k[i] - 1.0, 1e-8);
        pi = rhoi * kminus1 * (Ei - 0.5 * ui * ui);
        pi = MAXD(pi, p_floor);
        Efi = pi / (rhoi * kminus1) + 0.5 * ui * ui;
        a2 = MAXD(k[i] * pi / rhoi, a2_floor);

        rho[i] = rhoi;
        u[i] = ui;
        p[i] = pi;
        E_final[i] = Efi;
        a[i] = sqrt(a2);
    }
}

static void initial_riemann(double* U, const double* A, PrimitiveState left, PrimitiveState right, const double* k, int nx) {
    int half = nx / 2;
    double kL = k[0];
    double kR = k[nx - 1];

    double left_state[NVAR] = {
        left.rho,
        left.rho * left.u,
        left.p / (kL - 1.0) + 0.5 * left.rho * left.u * left.u,
    };

    double right_state[NVAR] = {
        right.rho,
        right.rho * right.u,
        right.p / (kR - 1.0) + 0.5 * right.rho * right.u * right.u,
    };

    for (int i = 0; i < nx; ++i) {
        const double* s = (i < half) ? left_state : right_state;
        for (int v = 0; v < NVAR; ++v) {
            U[IDX(v, i, nx)] = s[v] * A[i];
        }
    }

    if ((nx % 2) != 0) {
        int mid = half;
        for (int v = 0; v < NVAR; ++v) {
            U[IDX(v, mid, nx)] = 0.5 * (left_state[v] + right_state[v]) * A[mid];
        }
    }
}

static void apply_atmospheric_outlet(double* U, const double* A, double* k, const SolverWorkspace* ws) {
    const int nx = ws->nx;
    const int i_in = nx - 4;
    const int nozzle_state = nx - 3;

    double A_in = MAXD(A[i_in], ws->area_floor);
    double rho_in = MAXD(U[IDX(0, i_in, nx)] / A_in, ws->rho_floor);
    double u_in = U[IDX(1, i_in, nx)] / (rho_in * A_in);
    double E_in = U[IDX(2, i_in, nx)] / (rho_in * A_in);
    double k_in = k[i_in];
    double p_in = rho_in * MAXD(k_in - 1.0, 1e-8) * (E_in - 0.5 * u_in * u_in);
    p_in = MAXD(p_in, ws->p_floor);
    double a_in = sqrt(MAXD(k_in * p_in / rho_in, ws->a2_floor));

    if (u_in >= a_in) {
        for (int i = nozzle_state; i < nx; ++i) {
            U[IDX(0, i, nx)] = U[IDX(0, i_in, nx)];
            U[IDX(1, i, nx)] = U[IDX(1, i_in, nx)];
            U[IDX(2, i, nx)] = U[IDX(2, i_in, nx)];
            k[i] = k[i_in];
        }
        enforce_positive_state(U, A, k, ws);
        return;
    }

    double rho_g;
    double u_g;
    double p_g;
    if (u_in >= 0.0) {
        rho_g = rho_in;
        u_g = u_in;
        p_g = p0;
    } else {
        rho_g = rho0;
        u_g = 0.0;
        p_g = p0;
    }

    double E_g = p_g / (k_in - 1.0) + 0.5 * rho_g * u_g * u_g;
    for (int i = nozzle_state; i < nx; ++i) {
        U[IDX(0, i, nx)] = rho_g * A[i];
        U[IDX(1, i, nx)] = rho_g * u_g * A[i];
        U[IDX(2, i, nx)] = E_g * A[i];
        k[i] = k_in;
    }

    enforce_positive_state(U, A, k, ws);
}

static void apply_nozzle_outlet(double* U, const double* A, double* k, const SolverWorkspace* ws) {
    const int nx = ws->nx;
    const int fictitious = nx - 4;
    const int nozzle_state = nx - 3;

    double A_f = MAXD(A[fictitious], ws->area_floor);
    double rho_f = MAXD(U[IDX(0, fictitious, nx)] / A_f, ws->rho_floor);
    double u_f = U[IDX(1, fictitious, nx)] / (rho_f * A_f);
    double E_f = U[IDX(2, fictitious, nx)] / (rho_f * A_f);
    double k_f = k[fictitious];
    double p_f = rho_f * MAXD(k_f - 1.0, 1e-8) * (E_f - 0.5 * u_f * u_f);

    if (!isfinite(rho_f) || !isfinite(u_f) || !isfinite(p_f) || rho_f <= 1e-12 || p_f <= MAXD(1.0, p0) || u_f <= 0.0) {
        apply_atmospheric_outlet(U, A, k, ws);
        return;
    }

    double R_f = gas_constant_from_k(k_f);
    double T_f = p_f / (rho_f * R_f);
    double a_f = sqrt(MAXD(k_f * p_f / rho_f, ws->a2_floor));
    double M_f = MAXD(u_f / a_f, 0.0);

    double T0_f = T_f * (1.0 + 0.5 * (k_f - 1.0) * M_f * M_f);
    double p0_f = p_f * pow(1.0 + 0.5 * (k_f - 1.0) * M_f * M_f, k_f / (k_f - 1.0));

    double Athroat = A2nozzle / epsnozzle;
    double area_ratio = A[fictitious] / Athroat;
    if (area_ratio <= 1.0) {
        apply_atmospheric_outlet(U, A, k, ws);
        return;
    }

    double M_s = solve_area_mach(area_ratio, k_f, 1);
    double T_s = T0_f / (1.0 + 0.5 * (k_f - 1.0) * M_s * M_s);
    double p_s = p0_f / pow(1.0 + 0.5 * (k_f - 1.0) * M_s * M_s, k_f / (k_f - 1.0));
    double rho_s = p_s / (R_f * T_s);
    double u_s = M_s * sqrt(k_f * R_f * T_s);

    if (!isfinite(rho_s) || !isfinite(u_s) || !isfinite(p_s) || rho_s <= 1e-12 || p_s <= 1.0) {
        apply_atmospheric_outlet(U, A, k, ws);
        return;
    }

    double E_s = p_s / (k_f - 1.0) + 0.5 * rho_s * u_s * u_s;
    for (int i = nozzle_state; i < nx; ++i) {
        U[IDX(0, i, nx)] = rho_s * A[i];
        U[IDX(1, i, nx)] = rho_s * u_s * A[i];
        U[IDX(2, i, nx)] = E_s * A[i];
        k[i] = k_f;
    }
}

static int riemann_bc(double* U, const double* A, double* k, const char* boundary_case, const SolverWorkspace* ws) {
    const int nx = ws->nx;

    int left_dst[3] = {2, 1, 0};
    int left_src[3] = {3, 4, 5};
    for (int j = 0; j < 3; ++j) {
        int d = left_dst[j];
        int s = left_src[j];
        U[IDX(0, d, nx)] = U[IDX(0, s, nx)];
        U[IDX(1, d, nx)] = -U[IDX(1, s, nx)];
        U[IDX(2, d, nx)] = U[IDX(2, s, nx)];
    }

    if (strcmp(boundary_case, "wall-atmosphere") != 0) {
        return 0;
    }

    apply_nozzle_outlet(U, A, k, ws);
    return 1;
}

static void invert3x3(const double M[9], double Minv[9]) {
    double det =
        M[0] * (M[4] * M[8] - M[5] * M[7]) -
        M[1] * (M[3] * M[8] - M[5] * M[6]) +
        M[2] * (M[3] * M[7] - M[4] * M[6]);

    det = regularize_denom(det, 1e-14);
    double invdet = 1.0 / det;

    Minv[0] = (M[4] * M[8] - M[5] * M[7]) * invdet;
    Minv[1] = (M[2] * M[7] - M[1] * M[8]) * invdet;
    Minv[2] = (M[1] * M[5] - M[2] * M[4]) * invdet;

    Minv[3] = (M[5] * M[6] - M[3] * M[8]) * invdet;
    Minv[4] = (M[0] * M[8] - M[2] * M[6]) * invdet;
    Minv[5] = (M[2] * M[3] - M[0] * M[5]) * invdet;

    Minv[6] = (M[3] * M[7] - M[4] * M[6]) * invdet;
    Minv[7] = (M[1] * M[6] - M[0] * M[7]) * invdet;
    Minv[8] = (M[0] * M[4] - M[1] * M[3]) * invdet;
}

static void mat3x3_vec3(const double M[9], const double x[3], double y[3]) {
    y[0] = M[0] * x[0] + M[1] * x[1] + M[2] * x[2];
    y[1] = M[3] * x[0] + M[4] * x[1] + M[5] * x[2];
    y[2] = M[6] * x[0] + M[7] * x[1] + M[8] * x[2];
}

static void weno5_reconstruct(const double* U, const double* A, const double* k,
                              double* UL, double* UR, double* khat, const SolverWorkspace* ws) {
    const int nx = ws->nx;
    const int m = ws->m;
    const double eps_weno = 1e-6;

    /* Reconstruct a chunk of interfaces from a shared chunk of nodes. */
    const int block_size = nx / omp_get_max_threads() + 1; // = 50 pt nx = 200
    // const int node_stride = block_size - 1; // = 49 pt nx = 200
    const int n_tiles = omp_get_max_threads();
    int tid;

    #pragma omp parallel for private(tid) schedule(static)
    
                                
    for (int tile = 0; tile < n_tiles; ++tile) {
        // fprintf(stderr, "Thread %d starting WENO reconstruction\n", tid);
        const int jbeg = tile * block_size;
        const int jend = MIND(nx, jbeg + block_size);
        const int node_beg = jbeg;
        const int node_end = jend;
        const int n_nodes = node_end - node_beg;

        double u_nodes[NVAR * block_size];

        for (int il = 0; il < n_nodes; ++il) {
            int ig = node_beg + il;
            u_nodes[IDX(0, il, block_size)] = U[IDX(0, ig, nx)];
            u_nodes[IDX(1, il, block_size)] = U[IDX(1, ig, nx)];
            u_nodes[IDX(2, il, block_size)] = U[IDX(2, ig, nx)];
        }

        for (int j = jbeg; j < jend; ++j) {
            const int jl = j - jbeg;
            const int im3l = jl;
            const int im2l = jl + 1;
            const int im1l = jl + 2;
            const int i0l = jl + 3;
            const int ip1l = jl + 4;
            const int ip2l = jl + 5;

            const int im1 = j + 2;
            const int i0 = j + 3;

            double A_im1 = MAXD(A[im1], ws->area_floor);
            double A_i = MAXD(A[i0], ws->area_floor);

            double rhoL = MAXD(U[IDX(0, im1, nx)] / A_im1, ws->rho_floor);
            double velL = U[IDX(1, im1, nx)] / (rhoL * A_im1);
            double EL = U[IDX(2, im1, nx)] / (rhoL * A_im1);
            double kL = k[im1];
            double pL = rhoL * MAXD(kL - 1.0, 1e-8) * (EL - 0.5 * velL * velL);
            pL = MAXD(pL, ws->p_floor);
            double HL = EL + pL / rhoL;

            double rhoR = MAXD(U[IDX(0, i0, nx)] / A_i, ws->rho_floor);
            double velR = U[IDX(1, i0, nx)] / (rhoR * A_i);
            double ER = U[IDX(2, i0, nx)] / (rhoR * A_i);
            double kR = k[i0];
            double pR = rhoR * MAXD(kR - 1.0, 1e-8) * (ER - 0.5 * velR * velR);
            pR = MAXD(pR, ws->p_floor);
            double HR = ER + pR / rhoR;

            double sL = sqrt(MAXD(rhoL, ws->rho_floor));
            double sR = sqrt(MAXD(rhoR, ws->rho_floor));
            double denom = regularize_denom(sL + sR, 1e-12);
            double inv_denom = 1.0 / denom;

            double rhohat = sL * sR;
            double uhat = (sL * velL + sR * velR) * inv_denom;
            double Hhat = (sL * HL + sR * HR) * inv_denom;
            double kh = (sL * kL + sR * kR) * inv_denom;

            double ahat2 = (kh - 1.0) * (Hhat - 0.5 * uhat * uhat);
            ahat2 = MAXD(ahat2, ws->a2_floor);
            double ahat = sqrt(ahat2);
            khat[j] = kh;

            double Pm[9];
            double Lm[9];

            Pm[0] = 1.0;
            Pm[1] = rhohat / (2.0 * ahat);
            Pm[2] = rhohat / (2.0 * ahat);

            Pm[3] = uhat;
            Pm[4] = rhohat * (uhat + ahat) / (2.0 * ahat);
            Pm[5] = rhohat * (uhat - ahat) / (2.0 * ahat);

            Pm[6] = 0.5 * uhat * uhat;
            Pm[7] = rhohat * (Hhat + uhat * ahat) / (2.0 * ahat);
            Pm[8] = rhohat * (Hhat - uhat * ahat) / (2.0 * ahat);

            invert3x3(Pm, Lm);

            double u_im3[3] = {
                u_nodes[IDX(0, im3l, node_stride)],
                u_nodes[IDX(1, im3l, node_stride)],
                u_nodes[IDX(2, im3l, node_stride)]
            };
            double u_im2[3] = {
                u_nodes[IDX(0, im2l, node_stride)],
                u_nodes[IDX(1, im2l, node_stride)],
                u_nodes[IDX(2, im2l, node_stride)]
            };
            double u_im1[3] = {
                u_nodes[IDX(0, im1l, node_stride)],
                u_nodes[IDX(1, im1l, node_stride)],
                u_nodes[IDX(2, im1l, node_stride)]
            };
            double u_i[3] = {
                u_nodes[IDX(0, i0l, node_stride)],
                u_nodes[IDX(1, i0l, node_stride)],
                u_nodes[IDX(2, i0l, node_stride)]
            };
            double u_ip1[3] = {
                u_nodes[IDX(0, ip1l, node_stride)],
                u_nodes[IDX(1, ip1l, node_stride)],
                u_nodes[IDX(2, ip1l, node_stride)]
            };
            double u_ip2[3] = {
                u_nodes[IDX(0, ip2l, node_stride)],
                u_nodes[IDX(1, ip2l, node_stride)],
                u_nodes[IDX(2, ip2l, node_stride)]
            };

            double wm_im3[3], wm_im2[3], wm_im1[3], wm_i[3], wm_ip1[3], wm_ip2[3];
            double w_uL[3], w_uR[3];

            mat3x3_vec3(Lm, u_im3, wm_im3);
            mat3x3_vec3(Lm, u_im2, wm_im2);
            mat3x3_vec3(Lm, u_im1, wm_im1);
            mat3x3_vec3(Lm, u_i, wm_i);
            mat3x3_vec3(Lm, u_ip1, wm_ip1);
            mat3x3_vec3(Lm, u_ip2, wm_ip2);

            for (int c = 0; c < 3; ++c) {
                double p0L = (1.0 / 3.0) * wm_im3[c] - (7.0 / 6.0) * wm_im2[c] + (11.0 / 6.0) * wm_im1[c];
                double p1L = -(1.0 / 6.0) * wm_im2[c] + (5.0 / 6.0) * wm_im1[c] + (1.0 / 3.0) * wm_i[c];
                double p2L = (1.0 / 3.0) * wm_im1[c] + (5.0 / 6.0) * wm_i[c] - (1.0 / 6.0) * wm_ip1[c];

                double dwL1 = wm_im3[c] - 2.0 * wm_im2[c] + wm_im1[c];
                double dwL2 = wm_im2[c] - 2.0 * wm_im1[c] + wm_i[c];
                double dwL3 = wm_im1[c] - 2.0 * wm_i[c] + wm_ip1[c];

                double beta1L = (13.0 / 12.0) * dwL1 * dwL1 +
                                0.25 * (wm_im3[c] - 4.0 * wm_im2[c] + 3.0 * wm_im1[c]) *
                                (wm_im3[c] - 4.0 * wm_im2[c] + 3.0 * wm_im1[c]);
                double beta2L = (13.0 / 12.0) * dwL2 * dwL2 +
                                0.25 * (wm_im2[c] - wm_i[c]) * (wm_im2[c] - wm_i[c]);
                double beta3L = (13.0 / 12.0) * dwL3 * dwL3 +
                                0.25 * (3.0 * wm_im1[c] - 4.0 * wm_i[c] + wm_ip1[c]) *
                                (3.0 * wm_im1[c] - 4.0 * wm_i[c] + wm_ip1[c]);

                double alpha1L = 0.1 / ((eps_weno + beta1L) * (eps_weno + beta1L));
                double alpha2L = 0.6 / ((eps_weno + beta2L) * (eps_weno + beta2L));
                double alpha3L = 0.3 / ((eps_weno + beta3L) * (eps_weno + beta3L));
                double sumL = alpha1L + alpha2L + alpha3L;
                double w1L = alpha1L / sumL;
                double w2L = alpha2L / sumL;
                double w3L = 1.0 - w1L - w2L;
                w_uL[c] = w1L * p0L + w2L * p1L + w3L * p2L;

                double p0R = -(1.0 / 6.0) * wm_im2[c] + (5.0 / 6.0) * wm_im1[c] + (1.0 / 3.0) * wm_i[c];
                double p1R = (1.0 / 3.0) * wm_im1[c] + (5.0 / 6.0) * wm_i[c] - (1.0 / 6.0) * wm_ip1[c];
                double p2R = (11.0 / 6.0) * wm_i[c] - (7.0 / 6.0) * wm_ip1[c] + (1.0 / 3.0) * wm_ip2[c];

                double dwR1 = wm_im2[c] - 2.0 * wm_im1[c] + wm_i[c];
                double dwR2 = wm_im1[c] - 2.0 * wm_i[c] + wm_ip1[c];
                double dwR3 = wm_i[c] - 2.0 * wm_ip1[c] + wm_ip2[c];

                double beta1R = (13.0 / 12.0) * dwR1 * dwR1 +
                                0.25 * (wm_im2[c] - 4.0 * wm_im1[c] + 3.0 * wm_i[c]) *
                                (wm_im2[c] - 4.0 * wm_im1[c] + 3.0 * wm_i[c]);
                double beta2R = (13.0 / 12.0) * dwR2 * dwR2 +
                                0.25 * (wm_im1[c] - wm_ip1[c]) * (wm_im1[c] - wm_ip1[c]);
                double beta3R = (13.0 / 12.0) * dwR3 * dwR3 +
                                0.25 * (3.0 * wm_i[c] - 4.0 * wm_ip1[c] + wm_ip2[c]) *
                                (3.0 * wm_i[c] - 4.0 * wm_ip1[c] + wm_ip2[c]);

                double alpha1R = 0.3 / ((eps_weno + beta1R) * (eps_weno + beta1R));
                double alpha2R = 0.6 / ((eps_weno + beta2R) * (eps_weno + beta2R));
                double alpha3R = 0.1 / ((eps_weno + beta3R) * (eps_weno + beta3R));
                double sumR = alpha1R + alpha2R + alpha3R;
                double w1R = alpha1R / sumR;
                double w2R = alpha2R / sumR;
                double w3R = 1.0 - w1R - w2R;
                w_uR[c] = w1R * p0R + w2R * p1R + w3R * p2R;
            }

            double ULv[3], URv[3];
            mat3x3_vec3(Pm, w_uL, ULv);
            mat3x3_vec3(Pm, w_uR, URv);

            UL[IDX(0, j, m)] = ULv[0];
            UL[IDX(1, j, m)] = ULv[1];
            UL[IDX(2, j, m)] = ULv[2];

            UR[IDX(0, j, m)] = URv[0];
            UR[IDX(1, j, m)] = URv[1];
            UR[IDX(2, j, m)] = URv[2];
        }
    }
}

static void euler_flux_general(const double* U, int nU, const double* A, int nA, const double* k,
                               double* flux, SolverWorkspace* ws) {
    primitives_general(U, nU, A, nA, k, ws->rho, ws->u, ws->p, ws->E, ws->a, ws);

    // #pragma omp parallel for if(nU > 128)
    for (int i = 0; i < nU; ++i) {
        double Ause = (nU != nA) ? 0.5 * (A[i + 2] + A[i + 3]) : A[i];
        Ause = MAXD(Ause, ws->area_floor);

        double u2_rho = ws->u[i] * ws->rho[i];
        double u_rho_E_plus_p = ws->u[i] * (ws->rho[i] * ws->E[i] + ws->p[i]);

        flux[IDX(0, i, nU)] = u2_rho * Ause;
        flux[IDX(1, i, nU)] = (u2_rho * ws->u[i] + ws->p[i]) * Ause;
        flux[IDX(2, i, nU)] = u_rho_E_plus_p * Ause;
    }
}

static double max_wave_speed_toro(const double* U, const double* A, const double* k,
                                  double* SL, double* SR, SolverWorkspace* ws) {
    weno5_reconstruct(U, A, k, ws->UL, ws->UR, ws->khat, ws);
    int m = ws->m;

    double max_abs = 0.0;
    for (int i = 0; i < m; ++i) {
    double AuseL = MAXD(0.5 * (A[i + 2] + A[i + 3]), ws->area_floor);
    double AuseR = AuseL;

    double rhoLi = MAXD(ws->UL[IDX(0, i, m)] / AuseL, ws->rho_floor);
    double rhoRi = MAXD(ws->UR[IDX(0, i, m)] / AuseR, ws->rho_floor);
    double uLi = ws->UL[IDX(1, i, m)] / (rhoLi * AuseL);
    double uRi = ws->UR[IDX(1, i, m)] / (rhoRi * AuseR);
    double ELi = ws->UL[IDX(2, i, m)] / (rhoLi * AuseL);
    double ERi = ws->UR[IDX(2, i, m)] / (rhoRi * AuseR);
    double pLi = rhoLi * MAXD(ws->khat[i] - 1.0, 1e-8) * (ELi - 0.5 * uLi * uLi);
    double pRi = rhoRi * MAXD(ws->khat[i] - 1.0, 1e-8) * (ERi - 0.5 * uRi * uRi);
    pLi = MAXD(pLi, ws->p_floor);
    pRi = MAXD(pRi, ws->p_floor);
    double aLi = sqrt(MAXD(ws->khat[i] * pLi / rhoLi, ws->a2_floor));
    double aRi = sqrt(MAXD(ws->khat[i] * pRi / rhoRi, ws->a2_floor));

        double kh = MAXD(ws->khat[i], 1.0 + 1e-6);

        double kh_m1 = kh - 1.0;
        double kh_p1 = kh + 1.0;
        double inv_2kh = 1.0 / (2.0 * kh);

        double gamma_exp = kh_m1 * inv_2kh;
        double power_exp = 2.0 * kh / kh_m1;

        double pL_gamma = pow(pLi, gamma_exp);
        double pR_gamma = pow(pRi, gamma_exp);

        double base_num = aLi + aRi - 0.5 * kh_m1 * (uRi - uLi);
        double base_den = aLi / pL_gamma + aRi / pR_gamma;
        double base = MAXD(base_num / regularize_denom(base_den, 1e-12), 1e-16);
        double pstarr = MAXD(pow(base, power_exp), ws->p_floor);

        double coeff = kh_p1 * inv_2kh;
        double pstarr_pL = pstarr / pLi;
        double pstarr_pR = pstarr / pRi;

        double qL_arg = MAXD(1.0 + coeff * (pstarr_pL - 1.0), 1.0);
        double qR_arg = MAXD(1.0 + coeff * (pstarr_pR - 1.0), 1.0);

        double qL = (pstarr <= pLi) ? 1.0 : sqrt(qL_arg);
        double qR = (pstarr <= pRi) ? 1.0 : sqrt(qR_arg);

        SL[i] = uLi - aLi * qL;
        SR[i] = uRi + aRi * qR;

        double aSL = fabs(SL[i]);
        double aSR = fabs(SR[i]);
        if (aSL > max_abs) {
            max_abs = aSL;
        }
        if (aSR > max_abs) {
            max_abs = aSR;
        }
    }
    return max_abs;
}

static void detect_troubled_from_recon(const double* UL, const double* UR, const double* A, const double* k,
                                       int* troubled, const SolverWorkspace* ws) {
    const int m = ws->m;

    for (int i = 0; i < m; ++i) {
        double Ause_l = MAXD(A[i + 2], ws->area_floor);
        double Ause_r = MAXD(A[i + 3], ws->area_floor);

        double rhoL_unsafe = UL[IDX(0, i, m)] / Ause_l;
        double rhoR_unsafe = UR[IDX(0, i, m)] / Ause_r;
        int tr = (rhoL_unsafe < 2.0 * ws->rho_floor) || (rhoR_unsafe < 2.0 * ws->rho_floor);

        double rhoL = MAXD(rhoL_unsafe, ws->rho_floor);
        double rhoR = MAXD(rhoR_unsafe, ws->rho_floor);

        double velL = UL[IDX(1, i, m)] / (rhoL * Ause_l);
        double velR = UR[IDX(1, i, m)] / (rhoR * Ause_r);
        double EL = UL[IDX(2, i, m)] / (rhoL * Ause_l);
        double ER = UR[IDX(2, i, m)] / (rhoR * Ause_r);

        double kLm1 = MAXD(k[i + 2] - 1.0, 1e-8);
        double kRm1 = MAXD(k[i + 3] - 1.0, 1e-8);

        double pL_unsafe = rhoL * kLm1 * (EL - 0.5 * velL * velL);
        double pR_unsafe = rhoR * kRm1 * (ER - 0.5 * velR * velR);

        if ((pL_unsafe < 2.0 * ws->p_floor) || (pR_unsafe < 2.0 * ws->p_floor)) {
            tr = 1;
        }

        double pL_safe = MAXD(pL_unsafe, ws->p_floor);
        double pR_safe = MAXD(pR_unsafe, ws->p_floor);
        double p_sum = pL_safe + pR_safe;
        double pressure_jump = fabs(pL_safe - pR_safe) / MAXD(0.5 * p_sum, ws->p_floor);
        if (pressure_jump > 0.5) {
            tr = 1;
        }
        troubled[i] = tr;
    }
}

static void hlle_flux_from_recon(const double* UL, const double* UR, const double* A, const double* khat,
                                 const double* SL, const double* SR, double* fm, double* fp,
                                 SolverWorkspace* ws) {
    const int m = ws->m;

    euler_flux_general(UL, m, A, ws->nx, khat, ws->fL, ws);
    euler_flux_general(UR, m, A, ws->nx, khat, ws->fR, ws);

    for (int i = 0; i < m; ++i) {
        double denom = regularize_denom(SR[i] - SL[i], 1e-12);
        for (int v = 0; v < NVAR; ++v) {
            ws->f[IDX(v, i, m)] =
                (SR[i] * ws->fL[IDX(v, i, m)] - SL[i] * ws->fR[IDX(v, i, m)] +
                 SL[i] * SR[i] * (UR[IDX(v, i, m)] - UL[IDX(v, i, m)])) /
                denom;
        }
    }

    for (int i = 0; i < m - 1; ++i) {
        for (int v = 0; v < NVAR; ++v) {
            fm[IDX(v, i, m - 1)] = ws->f[IDX(v, i, m)];
            fp[IDX(v, i, m - 1)] = ws->f[IDX(v, i + 1, m)];
        }
    }
}

static void hllc_flux(const double* U, const double* A, const double* k,
                      double* fm, double* fp, SolverWorkspace* ws) {
    const int m = ws->m;

    weno5_reconstruct(U, A, k, ws->UL, ws->UR, ws->khat, ws);

    euler_flux_general(ws->UL, m, A, ws->nx, ws->khat, ws->fL, ws);
    euler_flux_general(ws->UR, m, A, ws->nx, ws->khat, ws->fR, ws);

    max_wave_speed_toro(U, A, k, ws->SL, ws->SR, ws);

    for (int i = 0; i < m; ++i) {
        double Ause = MAXD(0.5 * (A[i + 2] + A[i + 3]), ws->area_floor);

        double rhoL = MAXD(ws->UL[IDX(0, i, m)] / Ause, ws->rho_floor);
        double rhoR = MAXD(ws->UR[IDX(0, i, m)] / Ause, ws->rho_floor);
        double uL = ws->UL[IDX(1, i, m)] / (rhoL * Ause);
        double uR = ws->UR[IDX(1, i, m)] / (rhoR * Ause);
        double EL = ws->UL[IDX(2, i, m)] / (rhoL * Ause);
        double ER = ws->UR[IDX(2, i, m)] / (rhoR * Ause);
        double pL = rhoL * MAXD(ws->khat[i] - 1.0, 1e-8) * (EL - 0.5 * uL * uL);
        double pR = rhoR * MAXD(ws->khat[i] - 1.0, 1e-8) * (ER - 0.5 * uR * uR);
        pL = MAXD(pL, ws->p_floor);
        pR = MAXD(pR, ws->p_floor);

        double denom_sstar = regularize_denom(rhoL * (ws->SL[i] - uL) - rhoR * (ws->SR[i] - uR), 1e-12);
        ws->Sstar[i] = (pR - pL + rhoL * uL * (ws->SL[i] - uL) - rhoR * uR * (ws->SR[i] - uR)) / denom_sstar;

        double denom_l = regularize_denom(ws->SL[i] - ws->Sstar[i], 1e-12);
        double denom_r = regularize_denom(ws->SR[i] - ws->Sstar[i], 1e-12);
        double denom_e_l = regularize_denom(rhoL * (ws->SL[i] - uL), 1e-12);
        double denom_e_r = regularize_denom(rhoR * (ws->SR[i] - uR), 1e-12);

        double UstarL[3];
        double UstarR[3];

        double facL = Ause * rhoL * (ws->SL[i] - uL) / denom_l;
        double facR = Ause * rhoR * (ws->SR[i] - uR) / denom_r;

        UstarL[0] = facL;
        UstarL[1] = facL * ws->Sstar[i];
        UstarL[2] = facL * (EL + (ws->Sstar[i] - uL) * (ws->Sstar[i] + pL / denom_e_l));

        UstarR[0] = facR;
        UstarR[1] = facR * ws->Sstar[i];
        UstarR[2] = facR * (ER + (ws->Sstar[i] - uR) * (ws->Sstar[i] + pR / denom_e_r));

        double fstarL[3];
        double fstarR[3];
        for (int v = 0; v < 3; ++v) {
            fstarL[v] = ws->fL[IDX(v, i, m)] + ws->SL[i] * (UstarL[v] - ws->UL[IDX(v, i, m)]);
            fstarR[v] = ws->fR[IDX(v, i, m)] + ws->SR[i] * (UstarR[v] - ws->UR[IDX(v, i, m)]);
        }

        if (ws->SL[i] >= 0.0) {
            for (int v = 0; v < 3; ++v) {
                ws->f[IDX(v, i, m)] = ws->fL[IDX(v, i, m)];
            }
        } else if (ws->Sstar[i] >= 0.0) {
            for (int v = 0; v < 3; ++v) {
                ws->f[IDX(v, i, m)] = fstarL[v];
            }
        } else if (ws->SR[i] > 0.0) {
            for (int v = 0; v < 3; ++v) {
                ws->f[IDX(v, i, m)] = fstarR[v];
            }
        } else {
            for (int v = 0; v < 3; ++v) {
                ws->f[IDX(v, i, m)] = ws->fR[IDX(v, i, m)];
            }
        }
    }

    for (int i = 0; i < m - 1; ++i) {
        for (int v = 0; v < 3; ++v) {
            ws->fm[IDX(v, i, m - 1)] = ws->f[IDX(v, i, m)];
            ws->fp[IDX(v, i, m - 1)] = ws->f[IDX(v, i + 1, m)];
        }
    }

    detect_troubled_from_recon(ws->UL, ws->UR, A, k, ws->troubled, ws);
    hlle_flux_from_recon(ws->UL, ws->UR, A, ws->khat, ws->SL, ws->SR, ws->fm_hlle, ws->fp_hlle, ws);

    for (int i = 0; i < m; ++i) {
        ws->theta[i] = ws->troubled[i] ? 0.0 : 1.0;
    }
    for (int i = 0; i < m; ++i) {
        if (ws->troubled[i]) {
            if (i > 0) {
                ws->theta[i - 1] = MIND(ws->theta[i - 1], 0.5);
            }
            if (i < m - 1) {
                ws->theta[i + 1] = MIND(ws->theta[i + 1], 0.5);
            }
        }
    }

    for (int i = 0; i < m - 1; ++i) {
        double theta_m = ws->theta[i];
        double theta_p = ws->theta[i + 1];
        double omt_m = 1.0 - theta_m;
        double omt_p = 1.0 - theta_p;
        for (int v = 0; v < 3; ++v) {
            fm[IDX(v, i, m - 1)] = theta_m * ws->fm[IDX(v, i, m - 1)] + omt_m * ws->fm_hlle[IDX(v, i, m - 1)];
            fp[IDX(v, i, m - 1)] = theta_p * ws->fp[IDX(v, i, m - 1)] + omt_p * ws->fp_hlle[IDX(v, i, m - 1)];
        }
    }
}

static void AP_map_batch(const double* A0, const double* dtrb, double* A1, double* P1, int n) {
    const double inv_sqrt_pi = 1.0 / sqrt(M_PI);
    for (int i = 0; i < n; ++i) {
        double r0 = sqrt(A0[i]) * inv_sqrt_pi;
        double r1 = r0 + dtrb[i];
        A1[i] = M_PI * r1 * r1;
        P1[i] = 2.0 * sqrt(A1[i] * M_PI);
    }
}

static void erosive_burning(const double* U, const double* A, const double* P, double dt, const double* k,
                            double* rb, SolverWorkspace* ws) {
    int nx = ws->nx;
    int n_real = ws->n_real;

    primitives_general(U, nx, A, nx, k, ws->rho, ws->u, ws->p, ws->E, ws->a, ws);

    for (int i = 0; i < n_real; ++i) {
        int gi = i + 3;
        double G = ws->rho[gi] * ws->u[gi];
        double Dh = 4.0 * A[gi] / P[gi];
        double rb0 = arb * pow(ws->p[gi], nrb);

        if (G > 1e-9 && Dh > 1e-9) {
            double re_scale = pow(G, 0.8) * pow(Dh, -0.2);
            double r_iter = rb0;
            for (int it = 0; it < 12; ++it) {
                double expo = -beta_er * rhosolid * r_iter / G;
                expo = MAXD(expo, -60.0);
                double re = alpha_er * re_scale * exp(expo);
                double r_new = rb0 + re;
                if (fabs(r_new - r_iter) <= 1e-6 * MAXD(1e-4, r_new)) {
                    r_iter = r_new;
                    break;
                }
                r_iter = r_new;
            }
            rb[i] = MAXD(0.0, r_iter);
        } else {
            rb[i] = rb0;
        }
    }

    (void)dt;
}

static int update_ignition_thermal_state(const double* U, const double* A, const double* P,
                                         ThermalState* ts, double dt, const double* k,
                                         SolverWorkspace* ws) {
    int nx = ws->nx;
    int n = ts->n_inner;

    primitives_general(U, nx, A, nx, k, ws->rho, ws->u, ws->p, ws->E, ws->a, ws);

    double alpha_s = Ksolid / (rhosolid * cpsolid);
    double coeff = 2.0 * sqrt(alpha_s) / (Ksolid * sqrt(M_PI));

    double t_old = ts->time;
    double t_new = t_old + dt;

    if (!ensure_thermal_history_capacity(ts, ts->hist_len + 1)) {
        return 0;
    }

    /*
     * Accumulate the thermal-history convolution in a cache-friendly order.
     * q_hist is stored as rows over i for each j, so iterating j outer / i inner
     * uses contiguous loads and computes sqrt-kernel only once per j.
     */
    for (int i = 0; i < n; ++i) {
        ws->tmp_nx[i] = 0.0;
    }
    if (ts->hist_len > 0) {
        for (int j = 0; j < ts->hist_len; ++j) {
            double dt_left = MAXD(t_new - ts->t_hist[j], 0.0);
            double dt_right = MAXD(t_new - ts->t_hist[j + 1], 0.0);
            double kernel = sqrt(dt_left) - sqrt(dt_right);
            const double* q_row = ts->q_hist + (size_t)j * (size_t)n;
            for (int i = 0; i < n; ++i) {
                ws->tmp_nx[i] += q_row[i] * kernel;
            }
        }
    }

    double* q_new_row = ts->q_hist + (size_t)ts->hist_len * (size_t)n;
    double sqrt_dt = sqrt(MAXD(dt, 1e-12));

    for (int i = 0; i < n; ++i) {
        int gi = i + 3;
        double R = ts->ignited[i] ? Rgas : Rgas_igniter;
        double mu = ts->ignited[i] ? mugas : mugas_igniter;
        double K = ts->ignited[i] ? Kgas : Kgas_igniter;
        double cp = ts->ignited[i] ? cpgas : cpgas_igniter;

        double Tg = ws->p[gi] / (ws->rho[gi] * R);
        double Dh = 4.0 * A[gi] / P[gi];
        double Re = ws->rho[gi] * fabs(ws->u[gi]) * Dh / mu;
        double Pr = cp * mu / K;
        double Nu = (Re > 2300.0) ? 0.023 * pow(Re, 0.8) * pow(Pr, 0.4) : 3.66;
        double htc = Nu * K / Dh;

        double history_term = ws->tmp_nx[i];
        double acoef = coeff * sqrt_dt * htc;
        double Ts_new = (Tp0 + coeff * history_term + acoef * Tg) / (1.0 + acoef);
        double q_new = htc * (Tg - Ts_new);

        ts->Ts[i] = Ts_new;
        if (Ts_new >= TSurf) {
            ts->ignited[i] = 1;
        }
        q_new_row[i] = q_new;
    }

    ts->hist_len += 1;
    ts->t_hist[ts->hist_len] = t_new;
    ts->time = t_new;
    return 1;
}

static void source_term(const double* U, const double* A, const double* P, const ThermalState* ts,
                        double dt, double t_local, const double* k, int erosive,
                        double* S, double* rb, double dx, SolverWorkspace* ws) {
    const int nx = ws->nx;
    const int n_updated = ws->n_updated;
    const int n_real = ws->n_real;

    primitives_general(U, nx, A, nx, k, ws->rho, ws->u, ws->p, ws->E, ws->a, ws);

    for (int i = 0; i < n_updated; ++i) {
        S[IDX(0, i, n_updated)] = 0.0;
        S[IDX(1, i, n_updated)] = 0.0;
        S[IDX(2, i, n_updated)] = 0.0;
    }

    for (int i = 0; i < n_updated; ++i) {
        double AifaceL = 0.5 * (A[i + 2] + A[i + 3]);
        double AifaceR = 0.5 * (A[i + 3] + A[i + 4]);
        double dAdx = (AifaceR - AifaceL) / dx;
        S[IDX(1, i, n_updated)] = ws->p[i + 3] * dAdx;
    }

    for (int i = 0; i < n_real; ++i) {
        rb[i] = 0.0;
    }

    if (erosive) {
        erosive_burning(U, A, P, dt, k, rb, ws);

        for (int i = 0; i < n_real; ++i) {
            if (!ts->ignited[i]) {
                rb[i] = 0.0;
            }
            double Sburn = P[i + 3];
            S[IDX(0, i, n_updated)] = rb[i] * Sburn * rhosolid;
            S[IDX(2, i, n_updated)] = rb[i] * Sburn * rhosolid * hreaction;
        }

        if (t_local < 0.35) {
            for (int i = 3; i < 7 && i < n_updated; ++i) {
                S[IDX(0, i, n_updated)] += mig / (4.0 * dx);
                S[IDX(1, i, n_updated)] += mig * vinj / (4.0 * dx);
                S[IDX(2, i, n_updated)] += mig * hig / (4.0 * dx);
            }
        }
    }
}

static double find_dt(const double* U, const double* A, double dx, double cfl, const double* k,
                      SolverWorkspace* ws) {
    double llam = max_wave_speed_toro(U, A, k, ws->SL, ws->SR, ws);
    if (!isfinite(llam) || llam <= 0.0) {
        return 1e-6;
    }
    return cfl * dx / llam;
}

static int ssprk45_step(const double* U, const double* A, const double* P,
                        double* Unext, double* Anew, double* Pnew,
                        ThermalState* ts, double dt, double dx, double t_local,
                        double* k, int erosive, const char* boundary_case,
                        SolverWorkspace* ws) {
    const int nx = ws->nx;
    const int n_updated = ws->n_updated;
    const int n_real = ws->n_real;
    const double inv_dx = -1.0 / dx;

    memcpy(ws->U1, U, (size_t)NVAR * (size_t)nx * sizeof(double));
    memcpy(ws->U2, U, (size_t)NVAR * (size_t)nx * sizeof(double));
    memcpy(ws->U3, U, (size_t)NVAR * (size_t)nx * sizeof(double));
    memcpy(ws->U4, U, (size_t)NVAR * (size_t)nx * sizeof(double));

    if (!update_ignition_thermal_state(U, A, P, ts, dt, k, ws)) {
        return 0;
    }

    double* Ustage = ws->U1;
    memcpy(Ustage, U, (size_t)NVAR * (size_t)nx * sizeof(double));
    if (!riemann_bc(Ustage, A, k, boundary_case, ws)) {
        return 0;
    }
    enforce_positive_state(Ustage, A, k, ws);
    hllc_flux(Ustage, A, k, ws->fm, ws->fp, ws);
    source_term(Ustage, A, P, ts, dt, t_local, k, erosive, ws->S1, ws->rb1, dx, ws);

    for (int i = 0; i < n_updated; ++i) {
        for (int v = 0; v < 3; ++v) {
            ws->k1[IDX(v, i, n_updated)] = inv_dx * (ws->fp[IDX(v, i, n_updated)] - ws->fm[IDX(v, i, n_updated)]) + ws->S1[IDX(v, i, n_updated)];
            ws->U1[IDX(v, i + 3, nx)] = U[IDX(v, i + 3, nx)] + 0.391752226571890 * dt * ws->k1[IDX(v, i, n_updated)];
        }
    }
    enforce_positive_state(ws->U1, A, k, ws);

    if (!riemann_bc(ws->U1, A, k, boundary_case, ws)) {
        return 0;
    }
    enforce_positive_state(ws->U1, A, k, ws);
    hllc_flux(ws->U1, A, k, ws->fm, ws->fp, ws);
    source_term(ws->U1, A, P, ts, dt, t_local, k, erosive, ws->S2, ws->rb2, dx, ws);

    for (int i = 0; i < n_updated; ++i) {
        for (int v = 0; v < 3; ++v) {
            ws->k2[IDX(v, i, n_updated)] = inv_dx * (ws->fp[IDX(v, i, n_updated)] - ws->fm[IDX(v, i, n_updated)]) + ws->S2[IDX(v, i, n_updated)];
            ws->U2[IDX(v, i + 3, nx)] =
                0.444370493651235 * U[IDX(v, i + 3, nx)] +
                0.555629506348765 * ws->U1[IDX(v, i + 3, nx)] +
                0.368410593050371 * dt * ws->k2[IDX(v, i, n_updated)];
        }
    }
    enforce_positive_state(ws->U2, A, k, ws);

    if (!riemann_bc(ws->U2, A, k, boundary_case, ws)) {
        return 0;
    }
    enforce_positive_state(ws->U2, A, k, ws);
    hllc_flux(ws->U2, A, k, ws->fm, ws->fp, ws);
    source_term(ws->U2, A, P, ts, dt, t_local, k, erosive, ws->S3, ws->rb3, dx, ws);

    for (int i = 0; i < n_updated; ++i) {
        for (int v = 0; v < 3; ++v) {
            ws->k3[IDX(v, i, n_updated)] = inv_dx * (ws->fp[IDX(v, i, n_updated)] - ws->fm[IDX(v, i, n_updated)]) + ws->S3[IDX(v, i, n_updated)];
            ws->U3[IDX(v, i + 3, nx)] =
                0.620101851488403 * U[IDX(v, i + 3, nx)] +
                0.379898148511597 * ws->U2[IDX(v, i + 3, nx)] +
                0.251891774271694 * dt * ws->k3[IDX(v, i, n_updated)];
        }
    }
    enforce_positive_state(ws->U3, A, k, ws);

    if (!riemann_bc(ws->U3, A, k, boundary_case, ws)) {
        return 0;
    }
    enforce_positive_state(ws->U3, A, k, ws);
    hllc_flux(ws->U3, A, k, ws->fm, ws->fp, ws);
    source_term(ws->U3, A, P, ts, dt, t_local, k, erosive, ws->S4, ws->rb4, dx, ws);

    for (int i = 0; i < n_updated; ++i) {
        for (int v = 0; v < 3; ++v) {
            ws->k4[IDX(v, i, n_updated)] = inv_dx * (ws->fp[IDX(v, i, n_updated)] - ws->fm[IDX(v, i, n_updated)]) + ws->S4[IDX(v, i, n_updated)];
            ws->U4[IDX(v, i + 3, nx)] =
                0.178079954393132 * U[IDX(v, i + 3, nx)] +
                0.821920045606868 * ws->U3[IDX(v, i + 3, nx)] +
                0.544974750228521 * dt * ws->k4[IDX(v, i, n_updated)];
        }
    }
    enforce_positive_state(ws->U4, A, k, ws);

    if (!riemann_bc(ws->U4, A, k, boundary_case, ws)) {
        return 0;
    }
    enforce_positive_state(ws->U4, A, k, ws);
    hllc_flux(ws->U4, A, k, ws->fm, ws->fp, ws);
    source_term(ws->U4, A, P, ts, dt, t_local, k, erosive, ws->S5, ws->rb5, dx, ws);

    for (int i = 0; i < n_updated; ++i) {
        for (int v = 0; v < 3; ++v) {
            ws->k5[IDX(v, i, n_updated)] = inv_dx * (ws->fp[IDX(v, i, n_updated)] - ws->fm[IDX(v, i, n_updated)]) + ws->S5[IDX(v, i, n_updated)];
            Unext[IDX(v, i + 3, nx)] =
                0.517231671970585 * ws->U2[IDX(v, i + 3, nx)] +
                0.096059710526147 * ws->U3[IDX(v, i + 3, nx)] +
                0.063692468666290 * dt * ws->k4[IDX(v, i, n_updated)] +
                0.386708617503268 * ws->U4[IDX(v, i + 3, nx)] +
                0.226007483236906 * dt * ws->k5[IDX(v, i, n_updated)];
        }
    }

    memcpy(Anew, A, (size_t)nx * sizeof(double));
    memcpy(Pnew, P, (size_t)nx * sizeof(double));

    for (int i = 0; i < n_real; ++i) {
        double rb_cycle =
            0.1468118760847865 * ws->rb1[i] +
            0.24848290944497606 * ws->rb2[i] +
            0.10425883033198079 * ws->rb3[i] +
            0.27443890090135015 * ws->rb4[i] +
            0.226007483236906 * ws->rb5[i];
        ws->tmp_nx[i] = dt * rb_cycle;
    }

    AP_map_batch(A + 3, ws->tmp_nx, Anew + 3, Pnew + 3, n_real);
    fill_geometry_ghosts(Anew, Pnew, nx);
    return 1;
}

static int write_final_profiles(const char* out_dir, int case_id, const double* U, const double* A, const double* P,
                                const double* k, int nx, double xdom, SolverWorkspace* ws) {
    char path[512];
    snprintf(path, sizeof(path), "%s/final_profiles_case%d.csv", out_dir, case_id);
    FILE* fp = fopen(path, "w");
    if (!fp) {
        return 0;
    }

    primitives_general(U, nx, A, nx, k, ws->rho, ws->u, ws->p, ws->E, ws->a, ws);

    fprintf(fp, "i,x,A,P,rho,u,p,E,a,M,k\n");
    for (int i = 0; i < nx; ++i) {
        double x = xdom * ((double)i / (double)(nx - 1));
        double M = ws->u[i] / regularize_denom(ws->a[i], 1e-12);
        fprintf(fp, "%d,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e\n",
            i, x, A[i], P[i], ws->rho[i], ws->u[i], ws->p[i], ws->E[i], ws->a[i], M, k[i]);
    }

    fclose(fp);
    return 1;
}

static int write_geometry_history(const char* out_dir, int case_id, const GeometryHistory* gh) {
    char path[512];
    snprintf(path, sizeof(path), "%s/geometry_history_case%d.csv", out_dir, case_id);

    FILE* fp = fopen(path, "w");
    if (!fp) {
        return 0;
    }

    fprintf(fp, "t");
    for (int i = 0; i < gh->nx; ++i) {
        fprintf(fp, ",A_%d", i);
    }
    for (int i = 0; i < gh->nx; ++i) {
        fprintf(fp, ",P_%d", i);
    }
    fprintf(fp, "\n");

    for (int n = 0; n < gh->len; ++n) {
        fprintf(fp, "%.16e", gh->t[n]);
        for (int i = 0; i < gh->nx; ++i) {
            fprintf(fp, ",%.16e", gh->A[(size_t)n * (size_t)gh->nx + (size_t)i]);
        }
        for (int i = 0; i < gh->nx; ++i) {
            fprintf(fp, ",%.16e", gh->P[(size_t)n * (size_t)gh->nx + (size_t)i]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
    return 1;
}

int main(void) {
#ifdef _OPENMP
    double exec_start = omp_get_wtime();
#else
    double exec_start = (double)clock() / (double)CLOCKS_PER_SEC;
#endif

    const int nx = 200;
    const double cfl = 0.9;
    const int erosive = 1;
    const int print_progress = 0;
    const int test_cases[] = {38};
    const int num_cases = (int)(sizeof(test_cases) / sizeof(test_cases[0]));

    SolverWorkspace ws;
    if (!alloc_workspace(&ws, nx)) {
        fprintf(stderr, "Failed to allocate solver workspace.\n");
        free_workspace(&ws);
        return 1;
    }

    double* Ulast = (double*)calloc((size_t)NVAR * (size_t)nx, sizeof(double));
    double* Unext = (double*)calloc((size_t)NVAR * (size_t)nx, sizeof(double));
    double* A = (double*)calloc((size_t)nx, sizeof(double));
    double* P = (double*)calloc((size_t)nx, sizeof(double));
    double* Anew = (double*)calloc((size_t)nx, sizeof(double));
    double* Pnew = (double*)calloc((size_t)nx, sizeof(double));
    double* kglobal = (double*)calloc((size_t)nx, sizeof(double));

    if (!Ulast || !Unext || !A || !P || !Anew || !Pnew || !kglobal) {
        fprintf(stderr, "Failed to allocate main arrays.\n");
        free(Ulast);
        free(Unext);
        free(A);
        free(P);
        free(Anew);
        free(Pnew);
        free(kglobal);
        free_workspace(&ws);
        return 1;
    }

    for (int tc_idx = 0; tc_idx < num_cases; ++tc_idx) {
        int case_id = test_cases[tc_idx];
        RiemannCase rc;
        if (!get_test_case(case_id, &rc)) {
            fprintf(stderr, "Unknown case id %d\n", case_id);
            continue;
        }

        double t = 0.0;
        double t_end = rc.t_end;
        double xdom = rc.xdom;
        double dx = xdom / (double)(nx - 6);

        for (int i = 0; i < nx; ++i) {
            A[i] = 1.0;
            P[i] = 2.0 * sqrt(M_PI);
            kglobal[i] = kigniter;
            for (int v = 0; v < 3; ++v) {
                Ulast[IDX(v, i, nx)] = 0.0;
                Unext[IDX(v, i, nx)] = 0.0;
            }
        }

        fill_geometry_ghosts(A, P, nx);

        ThermalState thermal_state;
        if (!init_thermal_state(&thermal_state, nx)) {
            fprintf(stderr, "Failed to init thermal state.\n");
            continue;
        }

        GeometryHistory gh = {0};
        gh.nx = nx;

        initial_riemann(Ulast, A, rc.left, rc.right, kglobal, nx);
        if (!riemann_bc(Ulast, A, kglobal, rc.boundary_case, &ws)) {
            fprintf(stderr, "Unsupported boundary case %s\n", rc.boundary_case);
            free_thermal_state(&thermal_state);
            continue;
        }
        enforce_positive_state(Ulast, A, kglobal, &ws);

        if (!append_geometry_history(&gh, t, A, P)) {
            fprintf(stderr, "Failed to append geometry history.\n");
            free_thermal_state(&thermal_state);
            free_geometry_history(&gh);
            continue;
        }

        int nsteps = 0;
        while (t < t_end) {
            double dt = find_dt(Ulast, A, dx, cfl, kglobal, &ws);

            memcpy(Unext, Ulast, (size_t)NVAR * (size_t)nx * sizeof(double));
            if (!ssprk45_step(Ulast, A, P, Unext, Anew, Pnew,
                              &thermal_state, dt, dx, t, kglobal,
                              erosive, rc.boundary_case, &ws)) {
                fprintf(stderr, "SSPRK45 failed at step %d.\n", nsteps);
                break;
            }

            for (int i = 3; i < nx - 4; ++i) {
                int li = i - 3;
                kglobal[i] = thermal_state.ignited[li] ? kgas : kigniter;
            }
            for (int i = nx - 4; i < nx; ++i) {
                kglobal[i] = kglobal[nx - 5];
            }

            if (!riemann_bc(Unext, Anew, kglobal, rc.boundary_case, &ws)) {
                fprintf(stderr, "Boundary condition failed at step %d.\n", nsteps);
                break;
            }
            enforce_positive_state(Unext, Anew, kglobal, &ws);

            memcpy(Ulast, Unext, (size_t)NVAR * (size_t)nx * sizeof(double));
            memcpy(A, Anew, (size_t)nx * sizeof(double));
            memcpy(P, Pnew, (size_t)nx * sizeof(double));

            t += dt;
            nsteps += 1;

            if (!append_geometry_history(&gh, t, A, P)) {
                fprintf(stderr, "Failed to append geometry history.\n");
                break;
            }

            if (print_progress) {
                printf("t=%.12e ", t);
            }
        }

        char out_dir[256];
        snprintf(out_dir, sizeof(out_dir), "cfl%.1f_c%d_%s_t%.15g_x%.15g_c", cfl, case_id, rc.boundary_case, t, xdom);
        if (MKDIR(out_dir) != 0 && errno != EEXIST) {
            fprintf(stderr, "Failed to create output dir %s\n", out_dir);
        }

        if (!write_final_profiles(out_dir, case_id, Ulast, A, P, kglobal, nx, xdom, &ws)) {
            fprintf(stderr, "Failed writing final profiles for case %d\n", case_id);
        }
        if (!write_geometry_history(out_dir, case_id, &gh)) {
            fprintf(stderr, "Failed writing geometry history for case %d\n", case_id);
        }

        free_thermal_state(&thermal_state);
        free_geometry_history(&gh);
    }

    free(Ulast);
    free(Unext);
    free(A);
    free(P);
    free(Anew);
    free(Pnew);
    free(kglobal);
    free_workspace(&ws);

#ifdef _OPENMP
    double exec_end = omp_get_wtime();
#else
    double exec_end = (double)clock() / (double)CLOCKS_PER_SEC;
#endif
    printf("\nExecution time: %.6f s\n", exec_end - exec_start);

    return 0;
}
