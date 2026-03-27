#include "riemann_test_cases.h"

int get_test_case(int case_id, RiemannCase* out_case) {
    if (!out_case) {
        return 0;
    }

    switch (case_id) {
        case 1:
            out_case->left = (PrimitiveState){1.0, 1.0, 0.75};
            out_case->right = (PrimitiveState){0.125, 0.1, 0.0};
            out_case->t_end = 0.2;
            out_case->xdom = 1.0;
            out_case->boundary_case = "Riemann";
            return 1;
        case 2:
            out_case->left = (PrimitiveState){1.0, 0.4, -2.0};
            out_case->right = (PrimitiveState){1.0, 0.4, 2.0};
            out_case->t_end = 0.15;
            out_case->xdom = 1.0;
            out_case->boundary_case = "Riemann";
            return 1;
        case 3:
            out_case->left = (PrimitiveState){1.0, 1000.0, 0.0};
            out_case->right = (PrimitiveState){1.0, 0.01, 0.0};
            out_case->t_end = 0.012;
            out_case->xdom = 1.0;
            out_case->boundary_case = "Riemann";
            return 1;
        case 4:
            out_case->left = (PrimitiveState){5.99924, 460.894, 19.5975};
            out_case->right = (PrimitiveState){5.99242, 46.0950, -6.19633};
            out_case->t_end = 0.035;
            out_case->xdom = 1.0;
            out_case->boundary_case = "Riemann";
            return 1;
        case 5:
            out_case->left = (PrimitiveState){1.0, 1000.0, -19.59745};
            out_case->right = (PrimitiveState){1.0, 0.01, -19.59745};
            out_case->t_end = 0.012;
            out_case->xdom = 1.0;
            out_case->boundary_case = "Riemann";
            return 1;
        case 6:
            out_case->left = (PrimitiveState){1.0, 1e5, 0.0};
            out_case->right = (PrimitiveState){0.125, 1e4, 0.0};
            out_case->t_end = 6e-4;
            out_case->xdom = 1.0;
            out_case->boundary_case = "Riemann";
            return 1;
        case 7:
            out_case->left = (PrimitiveState){1.0, 1.8, 1.0};
            out_case->right = (PrimitiveState){1.0, 1.8, 0.0};
            out_case->t_end = 5.0;
            out_case->xdom = 10.0;
            out_case->boundary_case = "Riemann-wall";
            return 1;
        case 8:
            out_case->left = (PrimitiveState){1.225, 1e5, 100.0};
            out_case->right = (PrimitiveState){1.225, 1e5, 0.0};
            out_case->t_end = 0.0010;
            out_case->xdom = 1.0;
            out_case->boundary_case = "Riemann-wall";
            return 1;
        case 38:
            out_case->left = (PrimitiveState){1.225, 101325.0, 0.0};
            out_case->right = (PrimitiveState){1.225, 101325.0, 0.0};
            out_case->t_end = 0.03;
            out_case->xdom = 8.5;
            out_case->boundary_case = "wall-atmosphere";
            return 1;
        default:
            return 0;
    }
}
