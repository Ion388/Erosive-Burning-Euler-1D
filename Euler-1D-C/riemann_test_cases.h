#ifndef RIEMANN_TEST_CASES_H
#define RIEMANN_TEST_CASES_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double rho;
    double p;
    double u;
} PrimitiveState;

typedef struct {
    PrimitiveState left;
    PrimitiveState right;
    double t_end;
    double xdom;
    const char* boundary_case;
} RiemannCase;

int get_test_case(int case_id, RiemannCase* out_case);

#ifdef __cplusplus
}
#endif

#endif
