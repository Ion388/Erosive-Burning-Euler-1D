program rocket_tester
	use input_vegae_mod
	use riemann_test_cases_mod
	use omp_lib, only: omp_get_num_procs, omp_get_thread_limit, omp_get_max_threads, omp_set_num_threads, omp_set_dynamic
	use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
	implicit none

	integer, parameter :: nvar = 3
	integer, parameter :: nx = 1000

	real(dp), parameter :: area_floor = 1.0e-12_dp
	real(dp), parameter :: a2_floor = 1.0e-10_dp
	real(dp), parameter :: eps_denom = 1.0e-12_dp
	integer, parameter :: geom_write_stride = 25

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

	type thermal_state_t
		real(dp), allocatable :: Ts(:)
		logical, allocatable :: ignited(:)
		real(dp), allocatable :: q_hist(:, :)
		real(dp), allocatable :: t_hist(:)
		integer :: n_hist = 0
		integer :: cap = 0
		real(dp) :: time = 0.0_dp
	end type thermal_state_t

	type(thermal_state_t) :: thermal

	call system_clock(clock_start, clock_rate, clock_max)
	call configure_openmp_runtime()

	call init_input_vegae()
	rho_floor = max(1.0e-9_dp, 1.0e-8_dp * rho0)
	p_floor = max(1.0_dp, 1.0e-8_dp * p0)

	cfl = 0.9_dp
	dt_print = 1.0e-1_dp
	erosive = .true.
	print_progress = .false.

	case_id = 38
	call test_case(case_id, left_initial, right_initial, t_end, xdom, boundary_case)

	dx = xdom / real(nx - 6, dp)
	do i = 1, nx
		xlist(i) = real(i - 1, dp) * xdom / real(nx - 1, dp)
	end do

	kglobal = kigniter
	A = 1.0_dp
	P = 2.0_dp * sqrt(acos(-1.0_dp))
	call fill_geometry_ghosts(A, P)

	call init_solid_thermal_state(nx, thermal)

	call initial_riemann(Ulast, A, left_initial, right_initial, kglobal)
	call riemann_bc(Ulast, A, kglobal, trim(boundary_case))
	call enforce_positive_state(Ulast, A, kglobal)

	write(geom_filename, '("rocket_geometry_history_case", I0, ".dat")') case_id
	open(newunit=geom_unit, file=trim(geom_filename), status='replace', action='write', iostat=ios)
	if (ios /= 0) stop 'Failed to open geometry history output file.'
	write(geom_unit, '(A)') '# t A(1..nx) P(1..nx)'
	write(geom_unit, *) 0.0_dp, A, P

	write(thermo_filename, '("rocket_state_history_case", I0, ".dat")') case_id
	open(newunit=thermo_unit, file=trim(thermo_filename), status='replace', action='write', iostat=ios)
	if (ios /= 0) stop 'Failed to open state history output file.'
	write(thermo_unit, '(A)') '# t x rho u p M A P'
	call write_state_snapshot(thermo_unit, 0.0_dp, Ulast, A, P, kglobal)

	t = 0.0_dp
	n = 0
	next_print_time = dt_print
	do while (t < t_end)
		dt = find_dt(Ulast, A, dx, cfl, kglobal)

		call ssprk45(Ulast, A, P, thermal, dt, dx, t, kglobal, erosive, Unext, Anew, Pnew)
		A = Anew
		P = Pnew
		do i = 4, nx - 4
			if (thermal%ignited(i - 3)) then
				kglobal(i) = gamma_gas
			else
				kglobal(i) = kigniter
			end if
		end do
		kglobal(nx - 3:nx) = kglobal(nx - 4)

		call riemann_bc(Unext, A, kglobal, trim(boundary_case))
		call enforce_positive_state(Unext, A, kglobal)

		Ulast = Unext
		t = t + dt
		n = n + 1

		do while (t >= next_print_time)
			call write_state_snapshot(thermo_unit, t, Ulast, A, P, kglobal)
			write(geom_unit, *) t, A, P
			next_print_time = next_print_time + dt_print
		end do

		if (mod(n, geom_write_stride) == 0) write(geom_unit, *) t, A, P
		if (print_progress) write(*, '(A, ES14.6)') 't = ', t
	end do
	close(geom_unit)
	close(thermo_unit)

	call primitives(Ulast, A, kglobal, rho, u, p_prof, E, asnd)
	do i = 1, nx
		mach(i) = u(i) / max(asnd(i), 1.0e-14_dp)
	end do

	write(prof_filename, '("rocket_profiles_case", I0, ".dat")') case_id
	open(newunit=geom_unit, file=trim(prof_filename), status='replace', action='write', iostat=ios)
	if (ios /= 0) stop 'Failed to open profile output file.'
	write(geom_unit, '(A)') '# x rho u p E a A P k M'
		do i = 1, nx
			write(geom_unit, '(10(ES24.14E3,1X))') xlist(i), rho(i), u(i), p_prof(i), E(i), asnd(i), A(i), P(i), kglobal(i), mach(i)
	end do
	close(geom_unit)

		call system_clock(clock_end)
		if (clock_end >= clock_start) then
			runtime_sec = real(clock_end - clock_start, dp) / real(clock_rate, dp)
		else
			runtime_sec = real(clock_end + (clock_max - clock_start) + 1, dp) / real(clock_rate, dp)
		end if
		write(*, '(A, F12.6, A)') 'Total runtime: ', runtime_sec, ' s'

contains

	subroutine configure_openmp_runtime()
		integer :: n_procs, thread_limit, max_threads_before, max_threads_after
		integer :: requested_threads, env_status, env_length, read_status
		character(len=64) :: omp_threads_env

		n_procs = omp_get_num_procs()
		thread_limit = omp_get_thread_limit()
		max_threads_before = omp_get_max_threads()

		omp_threads_env = ''
		call get_environment_variable('OMP_NUM_THREADS', omp_threads_env, length=env_length, status=env_status)
		requested_threads = max_threads_before

		if (env_status == 0 .and. env_length > 0) then
			read(omp_threads_env(1:env_length), *, iostat=read_status) requested_threads
			if (read_status /= 0) requested_threads = max_threads_before
		else
			if (thread_limit > 0) then
				requested_threads = min(n_procs, thread_limit)
			else
				requested_threads = n_procs
			end if
		end if

		requested_threads = max(1, requested_threads)
		if (thread_limit > 0) requested_threads = min(requested_threads, thread_limit)

		call omp_set_dynamic(.false.)
		call omp_set_num_threads(requested_threads)
		max_threads_after = omp_get_max_threads()

		write(*, '(A, I0)') 'OpenMP processors available: ', n_procs
		if (env_status == 0 .and. env_length > 0) then
			write(*, '(A, A)') 'OpenMP OMP_NUM_THREADS env: ', trim(omp_threads_env(1:env_length))
		else
			write(*, '(A)') 'OpenMP OMP_NUM_THREADS env: <not set>'
		end if
		write(*, '(A, I0)') 'OpenMP thread limit: ', thread_limit
		write(*, '(A, I0)') 'OpenMP max threads before set: ', max_threads_before
		write(*, '(A, I0)') 'OpenMP max threads in run: ', max_threads_after
	end subroutine configure_openmp_runtime

		subroutine write_state_snapshot(unit_id, t_snap, U_snap, A_snap, P_snap, k_snap)
			integer, intent(in) :: unit_id
			real(dp), intent(in) :: t_snap
			real(dp), intent(in) :: U_snap(nvar, nx), A_snap(nx), P_snap(nx), k_snap(nx)
			real(dp) :: rho_loc(nx), u_loc(nx), p_loc(nx), e_loc(nx), a_loc(nx), mach_loc(nx)
			integer :: j

			call primitives(U_snap, A_snap, k_snap, rho_loc, u_loc, p_loc, e_loc, a_loc)
			do j = 1, nx
				mach_loc(j) = u_loc(j) / max(a_loc(j), 1.0e-14_dp)
				write(unit_id, '(8(ES24.14E3,1X))') t_snap, xlist(j), rho_loc(j), u_loc(j), p_loc(j), mach_loc(j), A_snap(j), P_snap(j)
			end do
			write(unit_id, '(A)') ''
		end subroutine write_state_snapshot

	pure real(dp) function regularize_denom(x) result(val)
		real(dp), intent(in) :: x
		if (abs(x) < eps_denom) then
			if (x >= 0.0_dp) then
				val = eps_denom
			else
				val = -eps_denom
			end if
		else
			val = x
		end if
	end function regularize_denom

	subroutine init_solid_thermal_state(nx_local, state)
		integer, intent(in) :: nx_local
		type(thermal_state_t), intent(inout) :: state
		integer :: n_inner

		n_inner = nx_local - 7
		allocate(state%Ts(n_inner), state%ignited(n_inner))
		state%Ts = Tp0
		state%ignited = .false.

		state%cap = 512
		allocate(state%q_hist(n_inner, state%cap))
		allocate(state%t_hist(state%cap + 1))
		state%q_hist = 0.0_dp
		state%t_hist = 0.0_dp
		state%n_hist = 0
		state%time = 0.0_dp
	end subroutine init_solid_thermal_state

	subroutine ensure_history_capacity(state)
		type(thermal_state_t), intent(inout) :: state
		real(dp), allocatable :: q_new(:, :), t_new(:)
		integer :: new_cap, n_inner

		if (state%n_hist < state%cap) return

		n_inner = size(state%Ts)
		new_cap = max(2 * state%cap, state%cap + 256)

		allocate(q_new(n_inner, new_cap))
		q_new = 0.0_dp
		q_new(:, 1:state%n_hist) = state%q_hist(:, 1:state%n_hist)
		call move_alloc(q_new, state%q_hist)

		allocate(t_new(new_cap + 1))
		t_new = 0.0_dp
		t_new(1:state%n_hist + 1) = state%t_hist(1:state%n_hist + 1)
		call move_alloc(t_new, state%t_hist)

		state%cap = new_cap
	end subroutine ensure_history_capacity

	subroutine initial_riemann(U, A_loc, left_state, right_state, k)
		real(dp), intent(out) :: U(nvar, nx)
		real(dp), intent(in) :: A_loc(nx), left_state(3), right_state(3), k(nx)
		real(dp) :: rhoL, pL, uL, rhoR, pR, uR, kL, kR, eL, eR
		integer :: j, mid

		rhoL = left_state(1)
		pL = left_state(2)
		uL = left_state(3)
		rhoR = right_state(1)
		pR = right_state(2)
		uR = right_state(3)

		kL = k(1)
		kR = k(nx)

		eL = pL / (kL - 1.0_dp) + 0.5_dp * rhoL * uL * uL
		eR = pR / (kR - 1.0_dp) + 0.5_dp * rhoR * uR * uR

		mid = nx / 2
		do j = 1, nx
			if (j <= mid) then
				U(1, j) = rhoL * A_loc(j)
				U(2, j) = rhoL * uL * A_loc(j)
				U(3, j) = eL * A_loc(j)
			else
				U(1, j) = rhoR * A_loc(j)
				U(2, j) = rhoR * uR * A_loc(j)
				U(3, j) = eR * A_loc(j)
			end if
		end do
	end subroutine initial_riemann

	subroutine fill_geometry_ghosts(A_loc, P_loc)
		real(dp), intent(inout) :: A_loc(nx), P_loc(nx)
		A_loc(1:3) = A_loc(4)
		A_loc(nx - 3:nx) = A_loc(nx - 4)
		P_loc(1:3) = P_loc(4)
		P_loc(nx - 3:nx) = P_loc(nx - 4)
	end subroutine fill_geometry_ghosts

	subroutine enforce_positive_state(U, A_loc, k)
		real(dp), intent(inout) :: U(nvar, nx)
		real(dp), intent(in) :: A_loc(nx), k(nx)
		real(dp) :: Ause, rho_val, rhoA, vel, Eint, pval, kminus1
		integer :: j

		do j = 1, nx
			Ause = max(A_loc(j), area_floor)
			rho_val = max(U(1, j) / Ause, rho_floor)
			rhoA = rho_val * Ause
			vel = U(2, j) / max(rhoA, 1.0e-20_dp)
			Eint = U(3, j) / max(rhoA, 1.0e-20_dp)
			kminus1 = max(k(j) - 1.0_dp, 1.0e-8_dp)
			pval = rho_val * kminus1 * (Eint - 0.5_dp * vel * vel)
			pval = max(pval, p_floor)
			Eint = pval / (rho_val * kminus1) + 0.5_dp * vel * vel

			U(1, j) = rhoA
			U(2, j) = vel * rhoA
			U(3, j) = rho_val * Eint * Ause
		end do
	end subroutine enforce_positive_state

	subroutine primitives(Uin, Ain, kin, rho_out, vel, pval, Eout, a)
		real(dp), intent(in) :: Uin(:, :), Ain(:), kin(:)
		real(dp), intent(out) :: rho_out(size(Uin, 2)), vel(size(Uin, 2)), pval(size(Uin, 2)), Eout(size(Uin, 2)), a(size(Uin, 2))
		real(dp) :: Ause_j, kminus1, u2, a2
		integer :: nloc, j
		logical :: same_size

		nloc = size(Uin, 2)
		same_size = (size(Ain) == nloc)

		do j = 1, nloc
			if (same_size) then
				Ause_j = Ain(j)
			else
				Ause_j = 0.5_dp * (Ain(j + 2) + Ain(j + 3))
			end if
			Ause_j = max(Ause_j, area_floor)
			rho_out(j) = max(Uin(1, j) / Ause_j, rho_floor)
			vel(j) = Uin(2, j) / max(rho_out(j) * Ause_j, 1.0e-20_dp)
			Eout(j) = Uin(3, j) / max(rho_out(j) * Ause_j, 1.0e-20_dp)
			kminus1 = max(kin(j) - 1.0_dp, 1.0e-8_dp)
			u2 = vel(j) * vel(j)
			pval(j) = rho_out(j) * kminus1 * (Eout(j) - 0.5_dp * u2)
			pval(j) = max(pval(j), p_floor)
			Eout(j) = pval(j) / (rho_out(j) * kminus1) + 0.5_dp * u2
			a2 = max(kin(j) * pval(j) / rho_out(j), a2_floor)
			a(j) = sqrt(a2)
		end do
	end subroutine primitives

	real(dp) function gas_constant_from_k(k_local) result(Rval)
		real(dp), intent(in) :: k_local
		if (abs(k_local - gamma_gas) <= abs(k_local - kigniter)) then
			Rval = Rgas
		else
			Rval = Rgas_igniter
		end if
	end function gas_constant_from_k

	subroutine apply_atmospheric_outlet(U, A_loc, k)
		real(dp), intent(inout) :: U(nvar, nx), k(nx)
		real(dp), intent(in) :: A_loc(nx)
		integer, parameter :: fictitious_cell_idx = nx - 3
		integer, parameter :: nozzle_state_idx = nx - 2
		real(dp) :: rho_in, u_in, E_in, k_in, p_in, a_in
		real(dp) :: rho_g, u_g, p_g, E_g

		rho_in = max(U(1, fictitious_cell_idx) / max(A_loc(fictitious_cell_idx), area_floor), rho_floor)
		u_in = U(2, fictitious_cell_idx) / max(rho_in * A_loc(fictitious_cell_idx), 1.0e-20_dp)
		E_in = U(3, fictitious_cell_idx) / max(rho_in * A_loc(fictitious_cell_idx), 1.0e-20_dp)
		k_in = k(fictitious_cell_idx)
		p_in = rho_in * max(k_in - 1.0_dp, 1.0e-8_dp) * (E_in - 0.5_dp * u_in * u_in)
		p_in = max(p_in, p_floor)
		a_in = sqrt(max(k_in * p_in / rho_in, a2_floor))

		if (u_in >= a_in) then
			U(:, nozzle_state_idx:nx) = spread(U(:, fictitious_cell_idx), dim=2, ncopies=3)
			k(nozzle_state_idx:nx) = k(fictitious_cell_idx)
			call enforce_positive_state(U, A_loc, k)
			return
		end if

		if (u_in >= 0.0_dp) then
			rho_g = rho_in
			u_g = u_in
			p_g = p0
		else
			rho_g = rho0
			u_g = 0.0_dp
			p_g = p0
		end if

		E_g = p_g / (k_in - 1.0_dp) + 0.5_dp * rho_g * u_g * u_g
		U(1, nozzle_state_idx:nx) = rho_g * A_loc(nozzle_state_idx:nx)
		U(2, nozzle_state_idx:nx) = rho_g * u_g * A_loc(nozzle_state_idx:nx)
		U(3, nozzle_state_idx:nx) = E_g * A_loc(nozzle_state_idx:nx)
		k(nozzle_state_idx:nx) = k_in
		call enforce_positive_state(U, A_loc, k)
	end subroutine apply_atmospheric_outlet

	subroutine apply_nozzle_outlet(U, A_loc, k)
		real(dp), intent(inout) :: U(nvar, nx), k(nx)
		real(dp), intent(in) :: A_loc(nx)
		integer, parameter :: fictitious_cell_idx = nx - 3
		integer, parameter :: nozzle_state_idx = nx - 2
		real(dp) :: rho_f, u_f, E_f, k_f, p_f, R_f, T_f, a_f, M_f
		real(dp) :: T0_f, p0_f, Athroat, area_ratio
		real(dp) :: M_s, T_s, p_s, rho_s, u_s, E_s

		rho_f = max(U(1, fictitious_cell_idx) / max(A_loc(fictitious_cell_idx), area_floor), rho_floor)
		u_f = U(2, fictitious_cell_idx) / max(rho_f * A_loc(fictitious_cell_idx), 1.0e-20_dp)
		E_f = U(3, fictitious_cell_idx) / max(rho_f * A_loc(fictitious_cell_idx), 1.0e-20_dp)
		k_f = k(fictitious_cell_idx)
		p_f = rho_f * max(k_f - 1.0_dp, 1.0e-8_dp) * (E_f - 0.5_dp * u_f * u_f)

		if (.not.(ieee_is_finite(rho_f) .and. ieee_is_finite(u_f) .and. ieee_is_finite(p_f))) then
			call apply_atmospheric_outlet(U, A_loc, k)
			return
		end if
		if (rho_f <= 1.0e-12_dp .or. p_f <= max(1.0_dp, p0) .or. u_f <= 0.0_dp) then
			call apply_atmospheric_outlet(U, A_loc, k)
			return
		end if

		R_f = gas_constant_from_k(k_f)
		T_f = p_f / max(rho_f * R_f, 1.0e-20_dp)
		a_f = sqrt(max(k_f * p_f / rho_f, a2_floor))
		M_f = max(u_f / a_f, 0.0_dp)

		T0_f = T_f * (1.0_dp + 0.5_dp * (k_f - 1.0_dp) * M_f * M_f)
		p0_f = p_f * (1.0_dp + 0.5_dp * (k_f - 1.0_dp) * M_f * M_f) ** (k_f / (k_f - 1.0_dp))

		Athroat = A2nozzle / epsnozzle
		area_ratio = A_loc(fictitious_cell_idx) / Athroat
		if (area_ratio <= 1.0_dp) then
			call apply_atmospheric_outlet(U, A_loc, k)
			return
		end if

		M_s = solve_area_mach(area_ratio, k_f, .true.)
		T_s = T0_f / (1.0_dp + 0.5_dp * (k_f - 1.0_dp) * M_s * M_s)
		p_s = p0_f / (1.0_dp + 0.5_dp * (k_f - 1.0_dp) * M_s * M_s) ** (k_f / (k_f - 1.0_dp))
		rho_s = p_s / max(R_f * T_s, 1.0e-20_dp)
		u_s = M_s * sqrt(k_f * R_f * T_s)

		if (.not.(ieee_is_finite(rho_s) .and. ieee_is_finite(u_s) .and. ieee_is_finite(p_s))) then
			call apply_atmospheric_outlet(U, A_loc, k)
			return
		end if
		if (rho_s <= 1.0e-12_dp .or. p_s <= 1.0_dp) then
			call apply_atmospheric_outlet(U, A_loc, k)
			return
		end if

		E_s = p_s / (k_f - 1.0_dp) + 0.5_dp * rho_s * u_s * u_s
		U(1, nozzle_state_idx:nx) = rho_s * A_loc(nozzle_state_idx:nx)
		U(2, nozzle_state_idx:nx) = rho_s * u_s * A_loc(nozzle_state_idx:nx)
		U(3, nozzle_state_idx:nx) = E_s * A_loc(nozzle_state_idx:nx)
		k(nozzle_state_idx:nx) = k_f
	end subroutine apply_nozzle_outlet

	subroutine riemann_bc(U, A_loc, k, bcase)
		real(dp), intent(inout) :: U(nvar, nx), k(nx)
		real(dp), intent(in) :: A_loc(nx)
		character(len=*), intent(in) :: bcase

		U(:, 3) = U(:, 4)
		U(:, 2) = U(:, 5)
		U(:, 1) = U(:, 6)
		U(2, 3) = -U(2, 4)
		U(2, 2) = -U(2, 5)
		U(2, 1) = -U(2, 6)

		if (trim(bcase) /= 'wall-atmosphere') stop 'boundary_case must be wall-atmosphere.'
		call apply_nozzle_outlet(U, A_loc, k)
	end subroutine riemann_bc

	subroutine invert3x3(Ain, Ainv)
		real(dp), intent(in) :: Ain(3, 3)
		real(dp), intent(out) :: Ainv(3, 3)
		real(dp) :: det

		det = Ain(1,1) * (Ain(2,2) * Ain(3,3) - Ain(2,3) * Ain(3,2)) - &
					Ain(1,2) * (Ain(2,1) * Ain(3,3) - Ain(2,3) * Ain(3,1)) + &
					Ain(1,3) * (Ain(2,1) * Ain(3,2) - Ain(2,2) * Ain(3,1))

		Ainv(1,1) = (Ain(2,2) * Ain(3,3) - Ain(2,3) * Ain(3,2)) / det
		Ainv(1,2) = (Ain(1,3) * Ain(3,2) - Ain(1,2) * Ain(3,3)) / det
		Ainv(1,3) = (Ain(1,2) * Ain(2,3) - Ain(1,3) * Ain(2,2)) / det

		Ainv(2,1) = (Ain(2,3) * Ain(3,1) - Ain(2,1) * Ain(3,3)) / det
		Ainv(2,2) = (Ain(1,1) * Ain(3,3) - Ain(1,3) * Ain(3,1)) / det
		Ainv(2,3) = (Ain(1,3) * Ain(2,1) - Ain(1,1) * Ain(2,3)) / det

		Ainv(3,1) = (Ain(2,1) * Ain(3,2) - Ain(2,2) * Ain(3,1)) / det
		Ainv(3,2) = (Ain(1,2) * Ain(3,1) - Ain(1,1) * Ain(3,2)) / det
		Ainv(3,3) = (Ain(1,1) * Ain(2,2) - Ain(1,2) * Ain(2,1)) / det
	end subroutine invert3x3

	subroutine weno5_reconstruct(U, A_loc, k, UL, UR, eigenvals, khat_out)
		real(dp), intent(in) :: U(nvar, nx), A_loc(nx), k(nx)
		real(dp), intent(out) :: UL(nvar, nx - 5), UR(nvar, nx - 5), eigenvals(nvar, nx - 5), khat_out(nx - 5)
		integer :: max_threads_after, jbeg, jend, tile, n_tile
		integer :: n_ifaces, n_tiles, iface_group_size
		real(dp) :: eps_weno
		integer :: j

		! --- Thread-private stencil variables (declared here, privatised below) ---
		real(dp) :: uim3(3), uim2(3), uim1(3), ui(3), uip1(3), uip2(3)
		real(dp) :: rhoL, velL, EL, kL, pL, HL
		real(dp) :: rhoR, velR, ER, kR, pR, HR
		real(dp) :: sL, sR, denom, rhohat, uhat, Hhat, khat, ahat2, ahat
		real(dp) :: Pmat(3,3), Lmat(3,3), wuL(3), wuR(3), wm(6,3)
		real(dp) :: p0L, p1L, p2L, p0R, p1R, p2R
		real(dp) :: dw1, dw2, dw3, b1, b2, b3, a1, a2, a3, asum
		real(dp) :: A_im1, A_i, kLm1, kRm1
		integer :: r

		eps_weno = 1.0e-6_dp
		n_ifaces = nx - 5
		max_threads_after = omp_get_max_threads()
    	n_tiles = max(1, max_threads_after)
		! print number of tiles
		! write(*, '(A, I0)') 'Number of tiles: ', max_threads_after
    	iface_group_size = (n_ifaces + n_tiles - 1) / n_tiles

		!$omp parallel do default(shared) &
		!$omp   shared(U, A_loc, k, UL, UR, eigenvals, khat_out, n_ifaces, eps_weno, rho_floor, p_floor) &
		!$omp   private(jbeg, jend, n_tile, j, r, uim3, uim2, uim1, ui, uip1, uip2, &
		!$omp           rhoL, velL, EL, kL, pL, HL, rhoR, velR, ER, kR, pR, HR, &
		!$omp           sL, sR, denom, rhohat, uhat, Hhat, khat, ahat2, ahat, &
		!$omp           Pmat, Lmat, wuL, wuR, wm, &
		!$omp           p0L, p1L, p2L, p0R, p1R, p2R, &
		!$omp           dw1, dw2, dw3, b1, b2, b3, a1, a2, a3, asum, &
		!$omp           A_im1, A_i, kLm1, kRm1) &
		!$omp   schedule(dynamic)
		do tile = 1, n_tiles
			jbeg = 1 + (tile - 1) * iface_group_size
			jend = min(n_ifaces, jbeg + iface_group_size - 1)
			! write(*, '(A, I0, A, I0)') 'Tile ', tile, ': processing interfaces ', jbeg, ' to ', jend
			do j = jbeg, jend
				uim3 = U(:, j)
				uim2 = U(:, j + 1)
				uim1 = U(:, j + 2)
				ui = U(:, j + 3)
				uip1 = U(:, j + 4)
				uip2 = U(:, j + 5)

				A_im1 = max(A_loc(j + 2), area_floor)
				A_i = max(A_loc(j + 3), area_floor)

				rhoL = max(uim1(1) / A_im1, rho_floor)
				velL = uim1(2) / max(rhoL * A_im1, 1.0e-20_dp)
				EL = uim1(3) / max(rhoL * A_im1, 1.0e-20_dp)
				kL = k(j + 2)
				kLm1 = max(kL - 1.0_dp, 1.0e-8_dp)
				pL = max(rhoL * kLm1 * (EL - 0.5_dp * velL * velL), p_floor)
				HL = EL + pL / rhoL

				rhoR = max(ui(1) / A_i, rho_floor)
				velR = ui(2) / max(rhoR * A_i, 1.0e-20_dp)
				ER = ui(3) / max(rhoR * A_i, 1.0e-20_dp)
				kR = k(j + 3)
				kRm1 = max(kR - 1.0_dp, 1.0e-8_dp)
				pR = max(rhoR * kRm1 * (ER - 0.5_dp * velR * velR), p_floor)
				HR = ER + pR / rhoR

				sL = sqrt(max(rhoL, rho_floor))
				sR = sqrt(max(rhoR, rho_floor))
				denom = regularize_denom(sL + sR)

				rhohat = sL * sR
				uhat = (sL * velL + sR * velR) / denom
				Hhat = (sL * HL + sR * HR) / denom
				khat = (sL * kL + sR * kR) / denom
				ahat2 = max((khat - 1.0_dp) * (Hhat - 0.5_dp * uhat * uhat), a2_floor)
				ahat = sqrt(ahat2)

				Pmat(1,1) = 1.0_dp
				Pmat(1,2) = rhohat / (2.0_dp * ahat)
				Pmat(1,3) = rhohat / (2.0_dp * ahat)
				Pmat(2,1) = uhat
				Pmat(2,2) = rhohat * (uhat + ahat) / (2.0_dp * ahat)
				Pmat(2,3) = rhohat * (uhat - ahat) / (2.0_dp * ahat)
				Pmat(3,1) = 0.5_dp * uhat * uhat
				Pmat(3,2) = rhohat * (Hhat + uhat * ahat) / (2.0_dp * ahat)
				Pmat(3,3) = rhohat * (Hhat - uhat * ahat) / (2.0_dp * ahat)

				call invert3x3(Pmat, Lmat)

				wm(1, :) = matmul(Lmat, uim3)
				wm(2, :) = matmul(Lmat, uim2)
				wm(3, :) = matmul(Lmat, uim1)
				wm(4, :) = matmul(Lmat, ui)
				wm(5, :) = matmul(Lmat, uip1)
				wm(6, :) = matmul(Lmat, uip2)

				do r = 1, 3
					p0L = (1.0_dp / 3.0_dp) * wm(1, r) - (7.0_dp / 6.0_dp) * wm(2, r) + (11.0_dp / 6.0_dp) * wm(3, r)
					p1L = -(1.0_dp / 6.0_dp) * wm(2, r) + (5.0_dp / 6.0_dp) * wm(3, r) + (1.0_dp / 3.0_dp) * wm(4, r)
					p2L = (1.0_dp / 3.0_dp) * wm(3, r) + (5.0_dp / 6.0_dp) * wm(4, r) - (1.0_dp / 6.0_dp) * wm(5, r)

					dw1 = wm(1, r) - 2.0_dp * wm(2, r) + wm(3, r)
					dw2 = wm(2, r) - 2.0_dp * wm(3, r) + wm(4, r)
					dw3 = wm(3, r) - 2.0_dp * wm(4, r) + wm(5, r)
					b1 = (13.0_dp / 12.0_dp) * dw1 * dw1 + 0.25_dp * (wm(1, r) - 4.0_dp * wm(2, r) + 3.0_dp * wm(3, r)) ** 2
					b2 = (13.0_dp / 12.0_dp) * dw2 * dw2 + 0.25_dp * (wm(2, r) - wm(4, r)) ** 2
					b3 = (13.0_dp / 12.0_dp) * dw3 * dw3 + 0.25_dp * (3.0_dp * wm(3, r) - 4.0_dp * wm(4, r) + wm(5, r)) ** 2
					a1 = (1.0_dp / 10.0_dp) / (eps_weno + b1) ** 2
					a2 = (3.0_dp / 5.0_dp) / (eps_weno + b2) ** 2
					a3 = (3.0_dp / 10.0_dp) / (eps_weno + b3) ** 2
					asum = a1 + a2 + a3
					a1 = a1 / asum
					a2 = a2 / asum
					a3 = 1.0_dp - a1 - a2
					wuL(r) = a1 * p0L + a2 * p1L + a3 * p2L

					p0R = -(1.0_dp / 6.0_dp) * wm(2, r) + (5.0_dp / 6.0_dp) * wm(3, r) + (1.0_dp / 3.0_dp) * wm(4, r)
					p1R = (1.0_dp / 3.0_dp) * wm(3, r) + (5.0_dp / 6.0_dp) * wm(4, r) - (1.0_dp / 6.0_dp) * wm(5, r)
					p2R = (11.0_dp / 6.0_dp) * wm(4, r) - (7.0_dp / 6.0_dp) * wm(5, r) + (1.0_dp / 3.0_dp) * wm(6, r)

					dw1 = wm(2, r) - 2.0_dp * wm(3, r) + wm(4, r)
					dw2 = wm(3, r) - 2.0_dp * wm(4, r) + wm(5, r)
					dw3 = wm(4, r) - 2.0_dp * wm(5, r) + wm(6, r)
					b1 = (13.0_dp / 12.0_dp) * dw1 * dw1 + 0.25_dp * (wm(2, r) - 4.0_dp * wm(3, r) + 3.0_dp * wm(4, r)) ** 2
					b2 = (13.0_dp / 12.0_dp) * dw2 * dw2 + 0.25_dp * (wm(3, r) - wm(5, r)) ** 2
					b3 = (13.0_dp / 12.0_dp) * dw3 * dw3 + 0.25_dp * (3.0_dp * wm(4, r) - 4.0_dp * wm(5, r) + wm(6, r)) ** 2
					a1 = (3.0_dp / 10.0_dp) / (eps_weno + b1) ** 2
					a2 = (3.0_dp / 5.0_dp) / (eps_weno + b2) ** 2
					a3 = (1.0_dp / 10.0_dp) / (eps_weno + b3) ** 2
					asum = a1 + a2 + a3
					a1 = a1 / asum
					a2 = a2 / asum
					a3 = 1.0_dp - a1 - a2
					wuR(r) = a1 * p0R + a2 * p1R + a3 * p2R
				end do

				UL(:, j) = matmul(Pmat, wuL)
				UR(:, j) = matmul(Pmat, wuR)
				eigenvals(:, j) = [uhat, uhat + ahat, uhat - ahat]
				khat_out(j) = khat
			end do
		end do
		!$omp end parallel do
	end subroutine weno5_reconstruct

	subroutine max_wave_speed_toro(U, A_loc, k, SL, SR, maxabs)
		real(dp), intent(in) :: U(nvar, nx), A_loc(nx), k(nx)
		real(dp), intent(out) :: SL(nx - 5), SR(nx - 5), maxabs
		integer :: n_ifaces
		real(dp) :: UL(3, nx - 5), UR(3, nx - 5), eig(3, nx - 5), khat(nx - 5)
		real(dp) :: rhoL_arr(nx - 5), velL_arr(nx - 5), pL_arr(nx - 5), eL_arr(nx - 5), aL_arr(nx - 5)
		real(dp) :: rhoR_arr(nx - 5), velR_arr(nx - 5), pR_arr(nx - 5), eR_arr(nx - 5), aR_arr(nx - 5)
		real(dp) :: khat_safe, kh_m1, kh_p1, inv2k, gexp, pexp, pL_gamma, pR_gamma
		real(dp) :: base_num, base_den, base, pstarr, coeff, qL, qR
		integer :: j

		call weno5_reconstruct(U, A_loc, k, UL, UR, eig, khat)
		call primitives(UL, A_loc, khat, rhoL_arr, velL_arr, pL_arr, eL_arr, aL_arr)
		call primitives(UR, A_loc, khat, rhoR_arr, velR_arr, pR_arr, eR_arr, aR_arr)

		n_ifaces = nx - 5
		maxabs = 0.0_dp

		do j = 1, n_ifaces
			pL_arr(j) = max(pL_arr(j), p_floor)
			pR_arr(j) = max(pR_arr(j), p_floor)
			khat_safe = max(khat(j), 1.0_dp + 1.0e-6_dp)
			kh_m1 = khat_safe - 1.0_dp
			kh_p1 = khat_safe + 1.0_dp
			inv2k = 1.0_dp / (2.0_dp * khat_safe)
			gexp = kh_m1 * inv2k
			pexp = 2.0_dp * khat_safe / kh_m1

			pL_gamma = pL_arr(j) ** gexp
			pR_gamma = pR_arr(j) ** gexp
			base_num = aL_arr(j) + aR_arr(j) - 0.5_dp * kh_m1 * (velR_arr(j) - velL_arr(j))
			base_den = aL_arr(j) / pL_gamma + aR_arr(j) / pR_gamma
			base = max(base_num / regularize_denom(base_den), 1.0e-16_dp)
			pstarr = max(base ** pexp, p_floor)

			coeff = kh_p1 * inv2k
			if (pstarr <= pL_arr(j)) then
				qL = 1.0_dp
			else
				qL = sqrt(max(1.0_dp + coeff * (pstarr / pL_arr(j) - 1.0_dp), 1.0_dp))
			end if
			if (pstarr <= pR_arr(j)) then
				qR = 1.0_dp
			else
				qR = sqrt(max(1.0_dp + coeff * (pstarr / pR_arr(j) - 1.0_dp), 1.0_dp))
			end if

			SL(j) = velL_arr(j) - aL_arr(j) * qL
			SR(j) = velR_arr(j) + aR_arr(j) * qR
			maxabs = max(maxabs, abs(SL(j)), abs(SR(j)))
		end do
	end subroutine max_wave_speed_toro

	subroutine euler_flux(Uin, A_loc, k, flux)
		real(dp), intent(in) :: Uin(:, :), A_loc(nx), k(:)
		real(dp), intent(out) :: flux(3, size(Uin, 2))
		real(dp) :: rho_arr(size(Uin, 2)), vel_arr(size(Uin, 2)), pval_arr(size(Uin, 2)), Eout_arr(size(Uin, 2)), a_arr(size(Uin, 2))
		real(dp) :: Ause_j
		integer :: nloc, j
		logical :: same_size

		nloc = size(Uin, 2)
		call primitives(Uin, A_loc, k, rho_arr, vel_arr, pval_arr, Eout_arr, a_arr)
		same_size = (nloc == nx)

		do j = 1, nloc
			if (same_size) then
				Ause_j = A_loc(j)
			else
				Ause_j = 0.5_dp * (A_loc(j + 2) + A_loc(j + 3))
			end if
			Ause_j = max(Ause_j, area_floor)
			flux(1, j) = rho_arr(j) * vel_arr(j) * Ause_j
			flux(2, j) = (rho_arr(j) * vel_arr(j) * vel_arr(j) + pval_arr(j)) * Ause_j
			flux(3, j) = vel_arr(j) * (rho_arr(j) * Eout_arr(j) + pval_arr(j)) * Ause_j
		end do
	end subroutine euler_flux

	subroutine detect_troubled_interfaces(U, A_loc, k, troubled)
		real(dp), intent(in) :: U(nvar, nx), A_loc(nx), k(nx)
		logical, intent(out) :: troubled(nx - 5)
		integer :: n_ifaces
		real(dp) :: UL(3, nx - 5), UR(3, nx - 5), eig(3, nx - 5), khat(nx - 5)
		real(dp) :: Ause_l, Ause_r, rhoL_u, rhoR_u, rhoL_val, rhoR_val
		real(dp) :: velL2, velR2, EL, ER, kLm1, kRm1, pL_u, pR_u, psum, jump
		integer :: j

		n_ifaces = nx - 5

		call weno5_reconstruct(U, A_loc, k, UL, UR, eig, khat)
		troubled = .false.

		do j = 1, n_ifaces
			Ause_l = max(A_loc(j + 2), area_floor)
			Ause_r = max(A_loc(j + 3), area_floor)
			rhoL_u = UL(1, j) / Ause_l
			rhoR_u = UR(1, j) / Ause_r

			if (rhoL_u < 2.0_dp * rho_floor .or. rhoR_u < 2.0_dp * rho_floor) then
				troubled(j) = .true.
			end if

			rhoL_val = max(rhoL_u, rho_floor)
			rhoR_val = max(rhoR_u, rho_floor)
			velL2 = (UL(2, j) / max(rhoL_val * Ause_l, 1.0e-20_dp)) ** 2
			velR2 = (UR(2, j) / max(rhoR_val * Ause_r, 1.0e-20_dp)) ** 2
			EL = UL(3, j) / max(rhoL_val * Ause_l, 1.0e-20_dp)
			ER = UR(3, j) / max(rhoR_val * Ause_r, 1.0e-20_dp)
			kLm1 = max(k(j + 2) - 1.0_dp, 1.0e-8_dp)
			kRm1 = max(k(j + 3) - 1.0_dp, 1.0e-8_dp)
			pL_u = rhoL_val * kLm1 * (EL - 0.5_dp * velL2)
			pR_u = rhoR_val * kRm1 * (ER - 0.5_dp * velR2)
			if (pL_u < 2.0_dp * p_floor .or. pR_u < 2.0_dp * p_floor) then
				troubled(j) = .true.
			end if

			psum = max(0.5_dp * (max(pL_u, p_floor) + max(pR_u, p_floor)), p_floor)
			jump = abs(max(pL_u, p_floor) - max(pR_u, p_floor)) / psum
			if (jump > 0.5_dp) troubled(j) = .true.
		end do
	end subroutine detect_troubled_interfaces

	subroutine hlle_flux(U, A_loc, k, fm, fp)
		real(dp), intent(in) :: U(nvar, nx), A_loc(nx), k(nx)
		real(dp), intent(out) :: fm(3, nx - 6), fp(3, nx - 6)
		integer :: n_ifaces
		real(dp) :: UL(3, nx - 5), UR(3, nx - 5), eig(3, nx - 5), khat(nx - 5)
		real(dp) :: fL(3, nx - 5), fR(3, nx - 5), fhlle(3, nx - 5)
		real(dp) :: SL(nx - 5), SR(nx - 5), maxabs, denom_val
		integer :: j

		n_ifaces = nx - 5

		call weno5_reconstruct(U, A_loc, k, UL, UR, eig, khat)
		call euler_flux(UL, A_loc, khat, fL)
		call euler_flux(UR, A_loc, khat, fR)
		call max_wave_speed_toro(U, A_loc, k, SL, SR, maxabs)

		do j = 1, n_ifaces
			denom_val = regularize_denom(SR(j) - SL(j))
			fhlle(:, j) = (SR(j) * fL(:, j) - SL(j) * fR(:, j) + SL(j) * SR(j) * (UR(:, j) - UL(:, j))) / denom_val
		end do

		fm = fhlle(:, 1:nx - 6)
		fp = fhlle(:, 2:nx - 5)
	end subroutine hlle_flux

	subroutine hllc_flux(U, A_loc, k, fm, fp)
		real(dp), intent(in) :: U(nvar, nx), A_loc(nx), k(nx)
		real(dp), intent(out) :: fm(3, nx - 6), fp(3, nx - 6)
		integer :: n_ifaces
		real(dp) :: UL(3, nx - 5), UR(3, nx - 5), eig(3, nx - 5), khat(nx - 5)
		real(dp) :: rhoL_arr(nx - 5), velL_arr(nx - 5), pL_arr(nx - 5), eL_arr(nx - 5), aL_arr(nx - 5)
		real(dp) :: rhoR_arr(nx - 5), velR_arr(nx - 5), pR_arr(nx - 5), eR_arr(nx - 5), aR_arr(nx - 5)
		real(dp) :: fL(3, nx - 5), fR(3, nx - 5), f(3, nx - 5)
		real(dp) :: SL(nx - 5), SR(nx - 5), Sstar(nx - 5), maxabs
		real(dp) :: UstarL(3, nx - 5), UstarR(3, nx - 5), fstarL(3, nx - 5), fstarR(3, nx - 5)
		real(dp) :: Ause, denom_sstar, denom_l, denom_r, denom_e_l, denom_e_r, denom_val
		real(dp) :: fhlle(3, nx - 5), fblend(3, nx - 5), theta(nx - 5)
		logical :: troubled(nx - 5)
		logical :: need_hlle
		integer :: j

		n_ifaces = nx - 5

		call weno5_reconstruct(U, A_loc, k, UL, UR, eig, khat)
		call primitives(UL, A_loc, khat, rhoL_arr, velL_arr, pL_arr, eL_arr, aL_arr)
		call primitives(UR, A_loc, khat, rhoR_arr, velR_arr, pR_arr, eR_arr, aR_arr)

		call euler_flux(UL, A_loc, khat, fL)
		call euler_flux(UR, A_loc, khat, fR)
		call max_wave_speed_toro(U, A_loc, k, SL, SR, maxabs)

		do j = 1, n_ifaces
			Ause = max(0.5_dp * (A_loc(j + 2) + A_loc(j + 3)), area_floor)
			denom_sstar = regularize_denom(rhoL_arr(j) * (SL(j) - velL_arr(j)) - rhoR_arr(j) * (SR(j) - velR_arr(j)))
			Sstar(j) = (pR_arr(j) - pL_arr(j) + rhoL_arr(j) * velL_arr(j) * (SL(j) - velL_arr(j)) - rhoR_arr(j) * velR_arr(j) * (SR(j) - velR_arr(j))) / denom_sstar

			denom_l = regularize_denom(SL(j) - Sstar(j))
			denom_r = regularize_denom(SR(j) - Sstar(j))
			denom_e_l = regularize_denom(rhoL_arr(j) * (SL(j) - velL_arr(j)))
			denom_e_r = regularize_denom(rhoR_arr(j) * (SR(j) - velR_arr(j)))

			UstarL(1, j) = Ause * rhoL_arr(j) * (SL(j) - velL_arr(j)) / denom_l
			UstarL(2, j) = UstarL(1, j) * Sstar(j)
			UstarL(3, j) = Ause * rhoL_arr(j) * (SL(j) - velL_arr(j)) / denom_l * (eL_arr(j) + (Sstar(j) - velL_arr(j)) * (Sstar(j) + pL_arr(j) / denom_e_l))

			UstarR(1, j) = Ause * rhoR_arr(j) * (SR(j) - velR_arr(j)) / denom_r
			UstarR(2, j) = UstarR(1, j) * Sstar(j)
			UstarR(3, j) = Ause * rhoR_arr(j) * (SR(j) - velR_arr(j)) / denom_r * (eR_arr(j) + (Sstar(j) - velR_arr(j)) * (Sstar(j) + pR_arr(j) / denom_e_r))

			fstarL(:, j) = fL(:, j) + SL(j) * (UstarL(:, j) - UL(:, j))
			fstarR(:, j) = fR(:, j) + SR(j) * (UstarR(:, j) - UR(:, j))

			if (SL(j) >= 0.0_dp) then
				f(:, j) = fL(:, j)
			else if (SL(j) < 0.0_dp .and. Sstar(j) >= 0.0_dp) then
				f(:, j) = fstarL(:, j)
			else if (SR(j) > 0.0_dp .and. Sstar(j) <= 0.0_dp) then
				f(:, j) = fstarR(:, j)
			else
				f(:, j) = fR(:, j)
			end if
		end do

		fm = f(:, 1:nx - 6)
		fp = f(:, 2:nx - 5)

		call detect_troubled_interfaces(U, A_loc, k, troubled)
		need_hlle = any(troubled)
		if (.not. need_hlle) return

		! --- Theta blending: sequential because of neighbor dependency ---
		theta = 1.0_dp
		do j = 1, n_ifaces
			if (troubled(j)) theta(j) = 0.0_dp
		end do
		do j = 1, n_ifaces
			if (troubled(j)) then
				if (j > 1) theta(j - 1) = min(theta(j - 1), 0.5_dp)
				if (j < n_ifaces) theta(j + 1) = min(theta(j + 1), 0.5_dp)
			end if
		end do

		do j = 1, n_ifaces
			if (theta(j) < 1.0_dp) then
				denom_val = regularize_denom(SR(j) - SL(j))
				fhlle(:, j) = (SR(j) * fL(:, j) - SL(j) * fR(:, j) + SL(j) * SR(j) * (UR(:, j) - UL(:, j))) / denom_val
				fblend(:, j) = theta(j) * f(:, j) + (1.0_dp - theta(j)) * fhlle(:, j)
			else
				fblend(:, j) = f(:, j)
			end if
		end do

		fm = fblend(:, 1:nx - 6)
		fp = fblend(:, 2:nx - 5)
	end subroutine hllc_flux

	subroutine erosive_burning(U, A_loc, P_loc, dt_local, k, rb)
		real(dp), intent(in) :: U(nvar, nx), A_loc(nx), P_loc(nx), dt_local, k(nx)
		real(dp), intent(out) :: rb(nx - 7)
		real(dp) :: rho_arr(nx), vel_arr(nx), pval_arr(nx), Eout_arr(nx), a_arr(nx)
		real(dp) :: G, Dh, rb0, r_iter, expo, re_scale, r_new
		logical :: any_g, any_dh
		integer :: j, it

		call primitives(U, A_loc, k, rho_arr, vel_arr, pval_arr, Eout_arr, a_arr)
		any_g = .false.
		any_dh = .false.
		do j = 1, nx - 7
			G = rho_arr(j + 3) * vel_arr(j + 3)
			Dh = 4.0_dp * A_loc(j + 3) / max(P_loc(j + 3), 1.0e-20_dp)
			if (G > 1.0e-9_dp) any_g = .true.
			if (Dh > 1.0e-9_dp) any_dh = .true.
		end do

		if (.not. (any_g .and. any_dh)) then
			do j = 1, nx - 7
				rb(j) = arb * (pval_arr(j + 3) ** nrb)
			end do
			return
		end if

		rb = 0.0_dp
		do j = 1, nx - 7
			G = rho_arr(j + 3) * vel_arr(j + 3)
			Dh = 4.0_dp * A_loc(j + 3) / max(P_loc(j + 3), 1.0e-20_dp)
			rb0 = arb * (pval_arr(j + 3) ** nrb)

			if (G > 1.0e-9_dp .and. Dh > 1.0e-9_dp) then
				re_scale = (G ** 0.8_dp) * (Dh ** (-0.2_dp))
				r_iter = rb0
				do it = 1, 12
					expo = -beta_er * rhosolid * r_iter / G
					expo = max(expo, -60.0_dp)
					r_new = rb0 + alpha_er * re_scale * exp(expo)
					if (abs(r_new - r_iter) <= 1.0e-6_dp * max(1.0e-4_dp, r_new)) exit
					r_iter = r_new
				end do
				rb(j) = max(0.0_dp, r_new)
			end if
		end do
	end subroutine erosive_burning

	subroutine update_ignition_thermal_state(U, A_loc, P_loc, state, dt_local, k)
		real(dp), intent(in) :: U(nvar, nx), A_loc(nx), P_loc(nx), dt_local, k(nx)
		type(thermal_state_t), intent(inout) :: state
		real(dp) :: rho_arr(nx), vel_arr(nx), pval_arr(nx), Eout_arr(nx), a_arr(nx)
		real(dp) :: rho_i, p_i, Rloc, muloc, Kloc, cploc
		real(dp) :: Tg, Dh, Re, Pr, Nu, htc, alpha_s, coeff
		real(dp) :: t_old, t_new, kernel, sqrt_dt, acoef
		real(dp) :: history_term(size(state%Ts)), Ts_new(size(state%Ts)), q_new(size(state%Ts))
		integer :: j, h, n_inner

		call primitives(U, A_loc, k, rho_arr, vel_arr, pval_arr, Eout_arr, a_arr)

		n_inner = size(state%Ts)
		history_term = 0.0_dp

		t_old = state%time
		t_new = t_old + dt_local

		! History term accumulation: each h-step updates ALL j simultaneously.
		! The kernel is the same for all j within one h, so parallelise over j inside.
		if (state%n_hist > 0) then
			do h = 1, state%n_hist
				kernel = sqrt(max(t_new - state%t_hist(h), 0.0_dp)) - sqrt(max(t_new - state%t_hist(h + 1), 0.0_dp))
				do j = 1, n_inner
					history_term(j) = history_term(j) + state%q_hist(j, h) * kernel
				end do
			end do
		end if

		alpha_s = Ksolid / (rhosolid * cpsolid)
		coeff = 2.0_dp * sqrt(alpha_s) / (Ksolid * sqrt(acos(-1.0_dp)))
		sqrt_dt = sqrt(max(dt_local, 1.0e-12_dp))

		do j = 1, n_inner
			rho_i = rho_arr(j + 3)
			p_i = pval_arr(j + 3)
			if (state%ignited(j)) then
				Rloc = Rgas
				muloc = mugas
				Kloc = Kgas
				cploc = cpgas
			else
				Rloc = Rgas_igniter
				muloc = mugas_igniter
				Kloc = Kgas_igniter
				cploc = cpgas_igniter
			end if

			Tg = p_i / max(rho_i * Rloc, 1.0e-20_dp)
			Dh = 4.0_dp * A_loc(j + 3) / max(P_loc(j + 3), 1.0e-20_dp)
			Re = rho_i * abs(vel_arr(j + 3)) * Dh / max(muloc, 1.0e-20_dp)
			Pr = cploc * muloc / max(Kloc, 1.0e-20_dp)
			if (Re > 2300.0_dp) then
				Nu = 0.023_dp * (Re ** 0.8_dp) * (Pr ** 0.4_dp)
			else
				Nu = 3.66_dp
			end if
			htc = Nu * Kloc / max(Dh, 1.0e-20_dp)

			acoef = coeff * sqrt_dt * htc
			Ts_new(j) = (Tp0 + coeff * history_term(j) + acoef * Tg) / (1.0_dp + acoef)
			q_new(j) = htc * (Tg - Ts_new(j))
		end do

		state%Ts = Ts_new
		do j = 1, size(state%ignited)
			state%ignited(j) = state%ignited(j) .or. (state%Ts(j) >= TSurf)
		end do

		call ensure_history_capacity(state)
		state%n_hist = state%n_hist + 1
		state%q_hist(:, state%n_hist) = q_new
		state%t_hist(state%n_hist + 1) = t_new
		state%time = t_new
	end subroutine update_ignition_thermal_state

	subroutine source_term(U, A_loc, P_loc, ignited, dx_local, t_local, k, erosive_on, S, rb)
		real(dp), intent(in) :: U(nvar, nx), A_loc(nx), P_loc(nx), dx_local, t_local, k(nx)
		logical, intent(in) :: ignited(nx - 7), erosive_on
		real(dp), intent(out) :: S(3, nx - 6), rb(nx - 7)
		real(dp) :: rho_arr(nx), vel_arr(nx), pval_arr(nx), Eout_arr(nx), a_arr(nx)
		real(dp) :: Ainterface(nx - 5), dAdx(nx - 6)
		integer :: j

		call primitives(U, A_loc, k, rho_arr, vel_arr, pval_arr, Eout_arr, a_arr)

		do j = 1, nx - 5
			Ainterface(j) = 0.5_dp * (A_loc(j + 3) + A_loc(j + 2))
		end do

		do j = 1, nx - 6
			dAdx(j) = (Ainterface(j + 1) - Ainterface(j)) / dx_local
		end do

		S = 0.0_dp
		do j = 1, nx - 6
			S(2, j) = pval_arr(j + 3) * dAdx(j)
		end do

		rb = 0.0_dp
		if (erosive_on) then
			call erosive_burning(U, A_loc, P_loc, dx_local, k, rb)
			do j = 1, nx - 7
				if (.not. ignited(j)) rb(j) = 0.0_dp
				S(1, j) = rb(j) * P_loc(j + 3) * rhosolid
				S(3, j) = rb(j) * P_loc(j + 3) * rhosolid * hreaction
			end do

			if (t_local < 0.35_dp) then
				if (nx - 6 >= 7) then
					  S(1, 4:7) = S(1, 4:7) + mig / (4.0_dp * dx_local)
					  S(2, 4:7) = S(2, 4:7) + mig * vinj / (4.0_dp * dx_local)
					  S(3, 4:7) = S(3, 4:7) + mig * hig / (4.0_dp * dx_local)
				end if
			end if
		end if
	end subroutine source_term

	real(dp) function find_dt(U, A_loc, dx_local, cfl_local, k) result(dt_out)
		real(dp), intent(in) :: U(nvar, nx), A_loc(nx), dx_local, cfl_local, k(nx)
		real(dp) :: SL(nx - 5), SR(nx - 5), maxabs
		call max_wave_speed_toro(U, A_loc, k, SL, SR, maxabs)
		if (.not. ieee_is_finite(maxabs)) then
			dt_out = 1.0e-6_dp
		else if (maxabs > 0.0_dp) then
			dt_out = cfl_local * dx_local / maxabs
		else
			dt_out = 1.0e-6_dp
		end if
	end function find_dt

	subroutine ap_map_vec(A0, dtrb, A1, P1)
		real(dp), intent(in) :: A0(:), dtrb(:)
		real(dp), intent(out) :: A1(size(A0)), P1(size(A0))
		real(dp) :: r0, r1
		integer :: j

		do j = 1, size(A0)
			r0 = sqrt(A0(j)) / sqrt(acos(-1.0_dp))
			r1 = r0 + dtrb(j)
			A1(j) = acos(-1.0_dp) * r1 * r1
			P1(j) = 2.0_dp * sqrt(A1(j) * acos(-1.0_dp))
		end do
	end subroutine ap_map_vec

	subroutine ssprk45(U, A_loc, P_loc, state, dt_local, dx_local, t_local, k, erosive_on, Unp1, Anew, Pnew)
		real(dp), intent(in) :: U(nvar, nx), A_loc(nx), P_loc(nx), dt_local, dx_local, t_local, k(nx)
		type(thermal_state_t), intent(inout) :: state
		logical, intent(in) :: erosive_on
		real(dp), intent(out) :: Unp1(nvar, nx), Anew(nx), Pnew(nx)

		real(dp) :: U0(nvar, nx), U1(nvar, nx), U2(nvar, nx), U3(nvar, nx), U4(nvar, nx), kwork(nx)
		real(dp) :: fm(3, nx - 6), fp(3, nx - 6), S(3, nx - 6)
		real(dp) :: k1(3, nx - 6), k2(3, nx - 6), k3(3, nx - 6), k4(3, nx - 6), k5(3, nx - 6)
		real(dp) :: rb1(nx - 7), rb2(nx - 7), rb3(nx - 7), rb4(nx - 7), rb5(nx - 7), rb_cycle(nx - 7)
		real(dp) :: Ainner(nx - 7), Pinner(nx - 7)
		integer :: j

		U0 = U
		U1 = U
		U2 = U
		U3 = U
		U4 = U
		Unp1 = U
		kwork = k

		call update_ignition_thermal_state(U0, A_loc, P_loc, state, dt_local, kwork)

		! --- Stage 1 ---
		call riemann_bc(U0, A_loc, kwork, 'wall-atmosphere')
		call enforce_positive_state(U0, A_loc, kwork)
		call hllc_flux(U0, A_loc, kwork, fm, fp)
		call source_term(U0, A_loc, P_loc, state%ignited, dx_local, t_local, kwork, erosive_on, S, rb1)
		k1 = -(fp - fm) / dx_local + S
		do j = 4, nx - 3
			U1(:, j) = U0(:, j) + 0.391752226571890_dp * dt_local * k1(:, j - 3)
		end do
		call enforce_positive_state(U1, A_loc, kwork)

		! --- Stage 2 ---
		call riemann_bc(U1, A_loc, kwork, 'wall-atmosphere')
		call enforce_positive_state(U1, A_loc, kwork)
		call hllc_flux(U1, A_loc, kwork, fm, fp)
		call source_term(U1, A_loc, P_loc, state%ignited, dx_local, t_local, kwork, erosive_on, S, rb2)
		k2 = -(fp - fm) / dx_local + S
		do j = 4, nx - 3
			U2(:, j) = 0.444370493651235_dp * U0(:, j) + &
						  0.555629506348765_dp * U1(:, j) + &
						  0.368410593050371_dp * dt_local * k2(:, j - 3)
		end do
		call enforce_positive_state(U2, A_loc, kwork)

		! --- Stage 3 ---
		call riemann_bc(U2, A_loc, kwork, 'wall-atmosphere')
		call enforce_positive_state(U2, A_loc, kwork)
		call hllc_flux(U2, A_loc, kwork, fm, fp)
		call source_term(U2, A_loc, P_loc, state%ignited, dx_local, t_local, kwork, erosive_on, S, rb3)
		k3 = -(fp - fm) / dx_local + S
		do j = 4, nx - 3
			U3(:, j) = 0.620101851488403_dp * U0(:, j) + &
						  0.379898148511597_dp * U2(:, j) + &
						  0.251891774271694_dp * dt_local * k3(:, j - 3)
		end do
		call enforce_positive_state(U3, A_loc, kwork)

		! --- Stage 4 ---
		call riemann_bc(U3, A_loc, kwork, 'wall-atmosphere')
		call enforce_positive_state(U3, A_loc, kwork)
		call hllc_flux(U3, A_loc, kwork, fm, fp)
		call source_term(U3, A_loc, P_loc, state%ignited, dx_local, t_local, kwork, erosive_on, S, rb4)
		k4 = -(fp - fm) / dx_local + S
		do j = 4, nx - 3
			U4(:, j) = 0.178079954393132_dp * U0(:, j) + &
						  0.821920045606868_dp * U3(:, j) + &
						  0.544974750228521_dp * dt_local * k4(:, j - 3)
		end do
		call enforce_positive_state(U4, A_loc, kwork)

		! --- Stage 5 ---
		call riemann_bc(U4, A_loc, kwork, 'wall-atmosphere')
		call enforce_positive_state(U4, A_loc, kwork)
		call hllc_flux(U4, A_loc, kwork, fm, fp)
		call source_term(U4, A_loc, P_loc, state%ignited, dx_local, t_local, kwork, erosive_on, S, rb5)
		k5 = -(fp - fm) / dx_local + S

		do j = 4, nx - 3
			Unp1(:, j) = 0.517231671970585_dp * U2(:, j) + 0.096059710526147_dp * U3(:, j) + &
							0.063692468666290_dp * dt_local * k4(:, j - 3) + 0.386708617503268_dp * U4(:, j) + 0.226007483236906_dp * dt_local * k5(:, j - 3)
		end do

		rb_cycle = 0.1468118760847865_dp * rb1 + 0.24848290944497606_dp * rb2 + 0.10425883033198079_dp * rb3 + &
							 0.27443890090135015_dp * rb4 + 0.226007483236906_dp * rb5

		Anew = A_loc
		Pnew = P_loc
		call ap_map_vec(A_loc(4:nx - 4), dt_local * rb_cycle, Ainner, Pinner)
		Anew(4:nx - 4) = Ainner
		Pnew(4:nx - 4) = Pinner
		call fill_geometry_ghosts(Anew, Pnew)
	end subroutine ssprk45

end program rocket_tester
