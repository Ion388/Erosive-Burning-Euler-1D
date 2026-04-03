program teste

    implicit none

    integer, parameter :: dp = selected_real_kind(15, 307)
    integer, parameter :: nvar = 3
    integer, parameter :: nx = 200

    real(dp), parameter :: area_floor = 1.0e-12_dp
    real(dp), parameter :: a2_floor = 1.0e-10_dp
    real(dp), parameter :: eps_denom = 1.0e-12_dp
    integer, parameter :: geom_write_stride = 25
    integer, parameter :: iface_group_size = 3  
    real(dp) :: rho_floor, p_floor
    real(dp) :: cfl, t, t_end, xdom, dx, dt, dt_print, next_print_time, runtime_sec
    real(dp) :: left_initial(3), right_initial(3)
    real(dp) :: Ulast(nvar, nx), Unext(nvar, nx)
    real(dp) :: A(nx), P(nx), Anew(nx), Pnew(nx)
    real(dp) :: kglobal(nx), xlist(nx)
    real(dp) :: rho(nx), u(nx), p_prof(nx), E(nx), asnd(nx)
    real(dp) :: mach(nx)
    character(len=32) :: boundary_case
    logical :: erosive, print_progress
    integer :: case_id, i, n, geom_unit, thermo_unit, ios
    integer :: clock_start, clock_end, clock_rate, clock_max
    character(len=256) :: geom_filename, prof_filename, thermo_filename
    integer :: j

    kglobal = 1.4_dp
    A = 1.0_dp
    P = 2.0_dp * sqrt(acos(-1.0_dp))

    do j = 1, nx
        Ulast(:, j) = [1.0_dp, 1.0_dp, 0.0_dp]
    end do

    call system_clock(clock_start, clock_rate, clock_max)

    do j = 1, 10000000
        call euler_flux(Ulast, A, kglobal, Unext)  ! Dummy call to ensure euler_flux is compiled    
    end do
    ! call euler_flux(Ulast, A, kglobal, Unext)  ! Dummy call to ensure euler_flux is compiled   

    call system_clock(clock_end)
		if (clock_end >= clock_start) then
			runtime_sec = real(clock_end - clock_start, dp) / real(clock_rate, dp)
		else
			runtime_sec = real(clock_end + (clock_max - clock_start) + 1, dp) / real(clock_rate, dp)
		end if
		write(*, '(A, F12.6, A)') 'Total runtime: ', runtime_sec, ' s' 

contains

    subroutine euler_flux(Uin, A_loc, k, flux)
        real(dp), intent(in) :: Uin(:, :), A_loc(nx), k(:)
        real(dp), intent(out) :: flux(3, size(Uin, 2))
        real(dp) :: rho(size(Uin, 2)), vel(size(Uin, 2)), pval(size(Uin, 2)), Eout(size(Uin, 2)), a(size(Uin, 2))
        real(dp) :: Ause(size(Uin, 2))
        integer :: nloc, j

        nloc = size(Uin, 2)

        if (nloc == nx) then
            Ause = A_loc
        else
            do j = 1, nloc
                Ause(j) = 0.5_dp * (A_loc(j + 2) + A_loc(j + 3))
            end do
        end if
        do j = 1, nloc
            Ause(j) = max(Ause(j), area_floor)
            flux(1, j) = rho(j) * vel(j) * Ause(j)
            flux(2, j) = (rho(j) * vel(j) * vel(j) + pval(j)) * Ause(j)
            flux(3, j) = vel(j) * (rho(j) * Eout(j) + pval(j)) * Ause(j)
        end do
    end subroutine euler_flux

end program teste