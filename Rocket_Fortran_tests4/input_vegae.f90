module input_vegae_mod
  implicit none
  integer, parameter :: dp = kind(1.0d0)

  real(dp), parameter :: Tp0 = 300.0_dp
  real(dp), parameter :: T2prop = 1825.0_dp
  real(dp), parameter :: TSurf = 900.0_dp
  real(dp), parameter :: T1prop = 3300.0_dp
  real(dp), parameter :: Ksolid = 0.1_dp
  real(dp), parameter :: cpsolid = 1230.0_dp

  real(dp), parameter :: arb = 3.0e-5_dp
  real(dp), parameter :: nrb = 0.35_dp
  real(dp), parameter :: rhosolid = 1810.0_dp

  real(dp), parameter :: mugas = 9.0e-5_dp
  real(dp), parameter :: Kgas = 0.587_dp
  real(dp), parameter :: cpgas = 2298.7_dp
  ! SCHIMBARE MAJORA: cp IA VALOARE SPECIFICATA SUB PROPRIETATILE DE TRANSPORT IN OUTPUTUL CEA. ERA LA FEL IN 0D
  ! real(dp), parameter :: cpgas = 2635.0_dp
  real(dp), parameter :: gamma_gas = 1.1364_dp
  real(dp), parameter :: Rgas = cpgas * (gamma_gas - 1.0_dp) / gamma_gas
  real(dp), parameter :: hreaction = 7.6e6_dp

  real(dp), parameter :: kigniter = 1.22_dp
  real(dp), parameter :: cpgas_igniter = 1900.0_dp
  real(dp), parameter :: Rgas_igniter = cpgas_igniter * (kigniter - 1.0_dp) / kigniter
  real(dp), parameter :: mugas_igniter = 8.0e-5_dp
  real(dp), parameter :: Kgas_igniter = 0.4_dp
  real(dp), parameter :: mig = 179.0_dp
  real(dp), parameter :: vinj = 1500.0_dp
  real(dp), parameter :: hig = 3.5e6_dp

  real(dp), parameter :: pamb = 101325.0_dp
  real(dp), parameter :: p0 = 101325.0_dp
  real(dp), parameter :: rho0 = 1.225_dp

  real(dp), parameter :: Lcase = 8.69_dp
  real(dp), parameter :: Rcase = 1.45_dp
  real(dp), parameter :: Vfree = 8.58_dp
  real(dp), parameter :: Ainitial = 1.0_dp
  real(dp), parameter :: Pinitial = 2.0_dp * sqrt(Ainitial * acos(-1.0_dp))
  real(dp), parameter :: A2nozzle = acos(-1.0_dp) * 1.76_dp * 1.76_dp / 4.0_dp
  real(dp), parameter :: epsnozzle = 16.0_dp

  real(dp), parameter :: beta_er = 53.0_dp
  real(dp), save :: alpha_er = 0.0_dp

contains

  pure real(dp) function area_mach_function(M, k) result(val)
    real(dp), intent(in) :: M, k
    val = (1.0_dp / M) * (((1.0_dp + 0.5_dp * (k - 1.0_dp) * M * M) / (0.5_dp * (k + 1.0_dp))) ** ((k + 1.0_dp) / (2.0_dp * (k - 1.0_dp))))
  end function area_mach_function

  real(dp) function solve_area_mach(area_ratio, k_local, supersonic) result(M)
    real(dp), intent(in) :: area_ratio, k_local
    logical, intent(in) :: supersonic
    real(dp) :: lo, hi, mid, f_mid
    integer :: iter

    if (area_ratio <= 1.0_dp) then
      M = 1.0_dp
      return
    end if

    if (supersonic) then
      lo = 1.0_dp + 1.0e-8_dp
      hi = max(2.0_dp, 0.5_dp * area_ratio + 1.0_dp)
      do while (area_mach_function(hi, k_local) < area_ratio .and. hi < 100.0_dp)
        hi = hi * 1.5_dp
      end do
    else
      lo = 1.0e-8_dp
      hi = 1.0_dp - 1.0e-8_dp
    end if

    do iter = 1, 80
      mid = 0.5_dp * (lo + hi)
      f_mid = area_mach_function(mid, k_local) - area_ratio
      if (supersonic) then
        if (f_mid > 0.0_dp) then
          hi = mid
        else
          lo = mid
        end if
      else
        if (f_mid > 0.0_dp) then
          lo = mid
        else
          hi = mid
        end if
      end if
    end do

    M = 0.5_dp * (lo + hi)
  end function solve_area_mach

  pure real(dp) function alpha_er_from_properties(cp_gas, mu_gas, K_loc, cs_solid, rho_p, T1, Ts, T2, Tp) result(val)
    real(dp), intent(in) :: cp_gas, mu_gas, K_loc, cs_solid, rho_p, T1, Ts, T2, Tp
    real(dp) :: cp_gas_local, cs_solid_local, Pr

    cp_gas_local = cp_gas * 0.239_dp / 1000.0_dp
    cs_solid_local = cs_solid * 0.239_dp / 1000.0_dp
    Pr = mu_gas * cp_gas_local * 4184.0_dp / K_loc

    if (abs(T2 - Tp) < 1.0e-14_dp) then
      val = 0.0_dp
    else
      val = 0.0288_dp * cp_gas_local * (mu_gas ** 0.2_dp) * (Pr ** (-2.0_dp / 3.0_dp)) / (rho_p * cs_solid_local) * ((T1 - Ts) / (T2 - Tp))
    end if
  end function alpha_er_from_properties

  subroutine init_input_vegae()
    alpha_er = alpha_er_from_properties(cpgas, mugas, Kgas, cpsolid, rhosolid, T1prop, TSurf, 1825.0_dp, 300.0_dp)
  end subroutine init_input_vegae

end module input_vegae_mod
