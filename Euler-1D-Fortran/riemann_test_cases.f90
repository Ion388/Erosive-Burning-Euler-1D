module riemann_test_cases_mod
  use input_vegae_mod, only: dp
  implicit none
contains

  subroutine test_case(case_id, left_state, right_state, t_end, xdom, boundary_case)
    integer, intent(in) :: case_id
    real(dp), intent(out) :: left_state(3), right_state(3), t_end, xdom
    character(len=*), intent(out) :: boundary_case

    select case (case_id)
    case (1)
      left_state = [1.0_dp, 1.0_dp, 0.75_dp]
      right_state = [0.125_dp, 0.1_dp, 0.0_dp]
      t_end = 0.2_dp
      xdom = 1.0_dp
      boundary_case = 'Riemann'
    case (2)
      left_state = [1.0_dp, 0.4_dp, -2.0_dp]
      right_state = [1.0_dp, 0.4_dp, 2.0_dp]
      t_end = 0.15_dp
      xdom = 1.0_dp
      boundary_case = 'Riemann'
    case (3)
      left_state = [1.0_dp, 1000.0_dp, 0.0_dp]
      right_state = [1.0_dp, 0.01_dp, 0.0_dp]
      t_end = 0.012_dp
      xdom = 1.0_dp
      boundary_case = 'Riemann'
    case (4)
      left_state = [5.99924_dp, 460.894_dp, 19.5975_dp]
      right_state = [5.99242_dp, 46.0950_dp, -6.19633_dp]
      t_end = 0.035_dp
      xdom = 1.0_dp
      boundary_case = 'Riemann'
    case (5)
      left_state = [1.0_dp, 1000.0_dp, -19.59745_dp]
      right_state = [1.0_dp, 0.01_dp, -19.59745_dp]
      t_end = 0.012_dp
      xdom = 1.0_dp
      boundary_case = 'Riemann'
    case (6)
      left_state = [1.0_dp, 1.0e5_dp, 0.0_dp]
      right_state = [0.125_dp, 1.0e4_dp, 0.0_dp]
      t_end = 6.0e-4_dp
      xdom = 1.0_dp
      boundary_case = 'Riemann'
    case (7)
      left_state = [1.0_dp, 1.8_dp, 1.0_dp]
      right_state = [1.0_dp, 1.8_dp, 0.0_dp]
      t_end = 5.0_dp
      xdom = 10.0_dp
      boundary_case = 'Riemann-wall'
    case (8)
      left_state = [1.225_dp, 1.0e5_dp, 100.0_dp]
      right_state = [1.225_dp, 1.0e5_dp, 0.0_dp]
      t_end = 0.0010_dp
      xdom = 1.0_dp
      boundary_case = 'Riemann-wall'
    case (38)
      left_state = [1.225_dp, 101325.0_dp, 0.0_dp]
      right_state = [1.225_dp, 101325.0_dp, 0.0_dp]
      t_end = 0.1_dp
      xdom = 8.5_dp
      boundary_case = 'wall-atmosphere'
    case default
      stop 'Unsupported test case.'
    end select
  end subroutine test_case

end module riemann_test_cases_mod
