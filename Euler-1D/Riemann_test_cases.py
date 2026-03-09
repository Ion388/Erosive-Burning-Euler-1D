def test_case(case): # (rho, p, u)
    if case == 1:
        return (1.0, 1.0, 0.75), (0.125, 0.1, 0.0), 0.2, 1.0, 'Riemann'
    elif case == 2:
        return (1.0, 0.4, -2.0), (1.0, 0.4, 2.0), 0.15, 1.0, 'Riemann'
    elif case == 3:
        return (1.0, 1000.0, 0.0), (1.0, 0.01, 0.0), 0.012, 1.0, 'Riemann'
    elif case == 4:
        return (5.99924, 460.894, 19.5975), (5.99242, 46.0950, -6.19633), 0.035, 1.0, 'Riemann'
    elif case == 5:
        return (1.0, 1000.0, -19.59745), (1.0, 0.01, -19.59745), 0.012, 1.0, 'Riemann'
    elif case == 6:
        return (1.0, 1e5, 0.0), (0.125, 1e4, 0.0), 6e-4, 1.0, 'Riemann'
    elif case == 7:
        return (1, 1.8, 1), (1, 1.8, 0), 5, 10.0, 'Riemann-wall'
    elif case == 8:
        return (1.225, 1e5, 100), (1.225, 1e5, 0.0), 0.0010, 1.0, 'Riemann-wall'