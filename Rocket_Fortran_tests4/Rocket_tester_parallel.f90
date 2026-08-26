program rocket_tester
	use input_vegae_mod
	use riemann_test_cases_mod
	use omp_lib, only: omp_get_num_procs, omp_get_thread_limit, omp_get_max_threads, &
		omp_set_num_threads, omp_set_dynamic, omp_get_wtime
	use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
	implicit none

	integer, parameter :: nvar = 3
	integer, parameter :: nx = 200
	! Diagnostic single-fluid closure. Set false only after adding a transported
	! composition variable and a thermodynamically consistent mixture EOS.
	logical, parameter :: use_single_gas_eos = .true.

	real(dp), parameter :: area_floor = 1.0e-12_dp
	real(dp), parameter :: a2_floor = 1.0e-10_dp
	real(dp), parameter :: eps_denom = 1.0e-12_dp
	real(dp), parameter :: source_cfl = 0.20_dp
	! Maximum fraction of one tabulated web interval traversed per full step.
	real(dp), parameter :: geometry_cfl = 0.02_dp
	real(dp), parameter :: mass_flux_floor = 1.0e-10_dp
	real(dp), parameter :: geometry_table_tol = 1.0e-10_dp
	real(dp), parameter :: reconstruction_floor_factor = 2.0_dp
	integer, parameter :: geom_write_stride = 25
	integer, parameter :: exit_write_stride = 10
	character(len=*), parameter :: grain_table_filename = 'grain_geometry_table.dat'

	type grain_geometry_table_t
		integer :: n_x = 0
		integer :: n_web = 0
		real(dp) :: web_max = 0.0_dp
		real(dp), allocatable :: x(:)
		real(dp), allocatable :: web(:)
		real(dp), allocatable :: A_port(:, :)
		real(dp), allocatable :: P_burn(:, :)
		real(dp), allocatable :: P_wet(:, :)
		integer, allocatable :: burned_out(:, :)
	end type grain_geometry_table_t

	real(dp) :: t0, prof_hllc, prof_source, prof_weno, prof_erosive, prof_validate, prof_primitives
	integer :: prof_call_hllc, prof_call_source, prof_call_weno, prof_call_erosive, prof_call_validate, prof_call_primitives
	integer :: reconstruction_limit_count, positivity_repair_count

	real(dp) :: rho_floor, p_floor
	real(dp) :: cfl, t, t_end, xdom, dx, dt, dt_print, next_print_time, runtime_sec
	real(dp) :: left_initial(3), right_initial(3)
	real(dp) :: Ulast(nvar, nx), Unext(nvar, nx)
	real(dp) :: A(nx), Pburn(nx), Pwet(nx)
	real(dp) :: Anew(nx), Pburnnew(nx), Pwetnew(nx)
	real(dp) :: web(nx - 7), webnew(nx - 7)
	real(dp) :: kglobal(nx), xlist(nx)
	real(dp) :: rho(nx), u(nx), p_prof(nx), E(nx), asnd(nx), rb_cycle(nx - 7), G_cycle(nx - 7)
	real(dp) :: mach(nx)
	real(dp) :: M_exit, T_exit, p_exit, rho_exit, u_exit, E_exit, thrust
	character(len=32) :: boundary_case
	logical :: erosive, print_progress
	integer :: case_id, i, n, geom_unit, thermo_unit, exit_unit, ios
	integer :: clock_start, clock_end, clock_rate, clock_max
	character(len=256) :: geom_filename, prof_filename, thermo_filename, exit_filename

	type thermal_state_t
		real(dp), allocatable :: Ts(:)
		logical, allocatable :: ignited(:)
		real(dp), allocatable :: q_hist(:, :)
		real(dp), allocatable :: t_hist(:)
		integer :: n_hist = 0
		integer :: cap = 0
		real(dp) :: time = 0.0_dp
		logical :: forced_ignition_reported = .false.
	end type thermal_state_t

	type(thermal_state_t) :: thermal
	type(grain_geometry_table_t) :: grain_table

	call system_clock(clock_start, clock_rate, clock_max)
	call configure_openmp_runtime()
	prof_hllc = 0.0_dp
	prof_source = 0.0_dp
	prof_weno = 0.0_dp
	prof_erosive = 0.0_dp
	prof_validate = 0.0_dp
	prof_primitives = 0.0_dp
	prof_call_hllc = 0
	prof_call_source = 0
	prof_call_weno = 0
	prof_call_erosive = 0
	prof_call_validate = 0
	prof_call_primitives = 0
	reconstruction_limit_count = 0
	positivity_repair_count = 0
	rb_cycle = 0.0_dp
	G_cycle = 0.0_dp
	M_exit = 0.0_dp
	T_exit = 0.0_dp
	p_exit = 0.0_dp
	rho_exit = 0.0_dp
	u_exit = 0.0_dp
	E_exit = 0.0_dp
	thrust = 0.0_dp


	call init_input_vegae()
	rho_floor = max(1.0e-9_dp, 1.0e-8_dp * rho0)
	p_floor = max(1.0_dp, 1.0e-8_dp * p0)
	! write(*, '(A, ES14.6)') 'p0 = ', p0

	cfl = 0.45_dp
	dt_print = 1.0e-3_dp
	erosive = .true.
	print_progress = .true.

	case_id = 38
	call test_case(case_id, left_initial, right_initial, t_end, xdom, boundary_case)

	dx = xdom / real(nx - 6, dp)
	do i = 1, nx
		xlist(i) = (real(i - 4, dp) + 0.5_dp) * dx
	end do

	if (use_single_gas_eos) then
		kglobal = gamma_gas
	else
		kglobal = kigniter
	end if
	call load_grain_geometry_table(grain_table_filename, nx - 7, xlist(4:nx - 4), grain_table)

	! The table is authoritative for the chamber port.  The input geometry is
	! retained only as the nozzle/ghost-cell template outside cells 4:nx-4.
	web = 0.0_dp
	A = Ainitial
	Pwet = Pinitial
	call fill_geometry_ghosts(A, Pwet)
	call geometry_from_web(web, grain_table, A, Pwet, Anew, Pburnnew, Pwetnew)
	A = Anew
	Pburn = Pburnnew
	Pwet = Pwetnew

	call init_solid_thermal_state(nx, thermal)

	call initial_riemann(Ulast, A, left_initial, right_initial, kglobal)
	call riemann_bc(Ulast, A, kglobal, trim(boundary_case))
	t0 = omp_get_wtime()
	call assert_admissible_state(Ulast, A, kglobal, 'initial state')
	prof_validate = prof_validate + (omp_get_wtime() - t0)
	prof_call_validate = prof_call_validate + 1

	write(geom_filename, '("rocket_geometry_history_case", I0, ".dat")') case_id
	open(newunit=geom_unit, file=trim(geom_filename), status='replace', action='write', iostat=ios)
	if (ios /= 0) stop 'Failed to open geometry history output file.'
	write(geom_unit, '(A)') '# t A(1..nx) P_wet(1..nx)'
	write(geom_unit, *) 0.0_dp, A, Pwet

	write(thermo_filename, '("rocket_state_history_case", I0, ".dat")') case_id
	open(newunit=thermo_unit, file=trim(thermo_filename), status='replace', action='write', iostat=ios)
	if (ios /= 0) stop 'Failed to open state history output file.'
	write(thermo_unit, '(A)') '# t x rho u p T M A P_wet rb G Ts ignited'
	call write_state_snapshot(thermo_unit, 0.0_dp, Ulast, A, Pwet, kglobal, thermal, rb_cycle, G_cycle)

	write(exit_filename, '("rocket_exit_history_case", I0, ".dat")') case_id
	open(newunit=exit_unit, file=trim(exit_filename), status='replace', action='write', iostat=ios)
	if (ios /= 0) stop 'Failed to open exit history output file.'
	write(exit_unit, '(A)') '# t M_exit T_exit p_exit rho_exit u_exit E_exit thrust'
	write(exit_unit, *) 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp


	t = 0.0_dp
	n = 0
	next_print_time = dt_print
	do while (t < t_end)
		if (t >= 0.35_dp .and. .not. thermal%forced_ignition_reported) then
			call report_and_force_ignition(thermal, t)
		end if
		dt = find_dt(Ulast, A, Pburn, Pwet, web, grain_table, dx, cfl, t, &
			kglobal, thermal%ignited, erosive)
		dt = min(dt, t_end - t)
		if (t < 0.35_dp) dt = min(dt, 0.35_dp - t)

		call ssprk45(Ulast, A, Pburn, Pwet, web, grain_table, thermal, dt, dx, t, &
			kglobal, erosive, Unext, Anew, Pburnnew, Pwetnew, webnew, rb_cycle, G_cycle)
		A = Anew
		Pburn = Pburnnew
		Pwet = Pwetnew
		web = webnew
		if (.not. use_single_gas_eos) then
			do i = 4, nx - 4
				if (thermal%ignited(i - 3)) then
					kglobal(i) = gamma_gas
				else
					kglobal(i) = kigniter
				end if
			end do
			kglobal(3) = kglobal(4)
			kglobal(2) = kglobal(5)
			kglobal(1) = kglobal(6)
			kglobal(nx - 3:nx) = kglobal(nx - 4)
		end if

		call riemann_bc(Unext, A, kglobal, trim(boundary_case))
		t0 = omp_get_wtime()
		call assert_admissible_state(Unext, A, kglobal, 'accepted full step')
		prof_validate = prof_validate + (omp_get_wtime() - t0)
		prof_call_validate = prof_call_validate + 1

		! call supersonic_properties(Unext, A, kglobal, M_exit, T_exit, p_exit, rho_exit, u_exit, E_exit)
		! write(*, '(A, ES14.6)') 'M_exit = ', M_exit

		Ulast = Unext
		t = t + dt
		n = n + 1

		do while (t >= next_print_time)
			call write_state_snapshot(thermo_unit, t, Ulast, A, Pwet, kglobal, thermal, rb_cycle, G_cycle)
			write(geom_unit, *) t, A, Pwet
			next_print_time = next_print_time + dt_print
		end do

		if (mod(n, geom_write_stride) == 0) write(geom_unit, *) t, A, Pwet
		if (mod(n, exit_write_stride) == 0) then
			call supersonic_properties(Unext, A, kglobal, M_exit, T_exit, p_exit, rho_exit, u_exit, E_exit, thrust)
			write(exit_unit, '(8(ES24.14E3,1X))') t, M_exit, T_exit, p_exit, rho_exit, u_exit, E_exit, thrust
		end if
		if (print_progress) write(*, '(A, ES14.6, A, ES14.6, A, ES14.6)') 't = ', t, '  dt = ', dt, '  M_exit = ', M_exit
	end do
	close(geom_unit)
	close(thermo_unit)
	close(exit_unit)

	call primitives(Ulast, A, kglobal, rho, u, p_prof, E, asnd)
	do i = 1, nx
		mach(i) = u(i) / max(asnd(i), 1.0e-14_dp)
	end do

	write(prof_filename, '("rocket_profiles_case", I0, ".dat")') case_id
	open(newunit=geom_unit, file=trim(prof_filename), status='replace', action='write', iostat=ios)
	if (ios /= 0) stop 'Failed to open profile output file.'
	write(geom_unit, '(A)') '# x rho u p E a A P_wet k M'
		do i = 1, nx
			write(geom_unit, '(10(ES24.14E3,1X))') xlist(i), rho(i), u(i), p_prof(i), &
				E(i), asnd(i), A(i), Pwet(i), kglobal(i), mach(i)
	end do
	close(geom_unit)

		call system_clock(clock_end)
		if (clock_end >= clock_start) then
			runtime_sec = real(clock_end - clock_start, dp) / real(clock_rate, dp)
		else
			runtime_sec = real(clock_end + (clock_max - clock_start) + 1, dp) / real(clock_rate, dp)
		end if
		write(*, '(A, F12.6, A)') 'Total runtime: ', runtime_sec, ' s'
		write(*, '(A, I0)') 'WENO interface-state positivity limits: ', reconstruction_limit_count
		write(*, '(A, I0)') 'Cell-average positivity repairs: ', positivity_repair_count

		! write(*, '(A, F12.6, A)') 'Total HLLC time: ', prof_hllc, ' s'
		! write(*, '(A, I0)') 'Total HLLC calls: ', prof_call_hllc
		! write(*, '(A, F12.6, A)') 'Total source term time: ', prof_source, ' s'
		! write(*, '(A, I0)') 'Total source term calls: ', prof_call_source
		! write(*, '(A, F12.6, A)') 'Total WENO time: ', prof_weno, ' s'
		! write(*, '(A, I0)') 'Total WENO calls: ', prof_call_weno
		! write(*, '(A, F12.6, A)') 'Total erosive burning time: ', prof_erosive, ' s'
		! write(*, '(A, I0)') 'Total erosive burning calls: ', prof_call_erosive
		! write(*, '(A, F12.6, A)') 'Total state validation time: ', prof_validate, ' s'
		! write(*, '(A, I0)') 'Total state validation calls: ', prof_call_validate
		! write(*, '(A, F12.6, A)') 'Total primitives time: ', prof_primitives, ' s'
		! write(*, '(A, I0)') 'Total primitives calls: ', prof_call_primitives

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

		subroutine write_state_snapshot(unit_id, t_snap, U_snap, A_snap, Pwet_snap, k_snap, thermal_snap, rb_cycle, G_cycle)
			integer, intent(in) :: unit_id
			real(dp), intent(in) :: t_snap
			real(dp), intent(in) :: U_snap(nvar, nx), A_snap(nx), Pwet_snap(nx), k_snap(nx)
			real(dp), intent(in) :: rb_cycle(nx - 7), G_cycle(nx - 7)
			type(thermal_state_t), intent(in) :: thermal_snap
			real(dp) :: rho_loc(nx), u_loc(nx), p_loc(nx), e_loc(nx), a_loc(nx), mach_loc(nx), T_loc(nx), R_loc(nx)
			real(dp) :: Ts_out, ignited_out
			integer :: j

		call primitives(U_snap, A_snap, k_snap, rho_loc, u_loc, p_loc, e_loc, a_loc)
		do j = 1, nx
			mach_loc(j) = u_loc(j) / max(a_loc(j), 1.0e-14_dp)
			R_loc(j) = gas_constant_from_k(k_snap(j))
			T_loc(j) = p_loc(j) / (rho_loc(j) * R_loc(j))
			if (j<=3 .or. j>=nx-3) then
				Ts_out = 0.0_dp
				ignited_out = 0.0_dp
				write(unit_id, '(13(ES24.14E3,1X))') t_snap, xlist(j), rho_loc(j), &
					u_loc(j), p_loc(j), T_loc(j), mach_loc(j), A_snap(j), Pwet_snap(j), &
					0.0_dp, 0.0_dp, Ts_out, ignited_out
			else
				Ts_out = thermal_snap%Ts(j - 3)
				ignited_out = merge(1.0_dp, 0.0_dp, thermal_snap%ignited(j - 3))
				write(unit_id, '(13(ES24.14E3,1X))') t_snap, xlist(j), rho_loc(j), &
					u_loc(j), p_loc(j), T_loc(j), mach_loc(j), A_snap(j), Pwet_snap(j), &
					rb_cycle(j-3), G_cycle(j-3), Ts_out, ignited_out
			end if
		end do
		write(unit_id, '(A)') ''
	end subroutine write_state_snapshot

	! Legacy exit-snapshot helper retained below for reference.
	! 	integer, intent(in) :: unit_id
	! 	real(dp), intent(in) :: t_snap
	! 	real(dp), intent(in) :: U_snap(nvar, nx), A_snap(nx), P_snap(nx), k_snap(nx)
	! 	real(dp) :: rho_loc(nx), u_loc(nx), p_loc(nx), e_loc(nx), a_loc(nx), mach_loc(nx)
	! 	integer :: j
	! 	real(dp) :: M_exit, T_exit, p_exit, rho_exit, u_exit, E_exit

	! 	call supersonic_properties(Unext, A, kglobal, M_exit, T_exit, p_exit, rho_exit, u_exit, E_exit)
	! 	write(unit_id, '(7(ES24.14E3,1X))') t_snap, M_exit, T_exit, p_exit, rho_exit, u_exit, E_exit
	! 	write(unit_id, '(A)') ''


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

	pure logical function geometry_values_match(a, b) result(match)
		real(dp), intent(in) :: a, b
		match = abs(a - b) <= geometry_table_tol * max(1.0_dp, abs(a), abs(b))
	end function geometry_values_match

	subroutine grain_table_fail(unit_id, message)
		integer, intent(in) :: unit_id
		character(len=*), intent(in) :: message
		integer :: close_status
		logical :: is_open

		write(*, '(A)') 'Invalid grain geometry table: '//trim(message)
		inquire(unit=unit_id, opened=is_open)
		if (is_open) close(unit_id, iostat=close_status)
		error stop 'Failed to load grain_geometry_table.dat.'
	end subroutine grain_table_fail

	subroutine load_grain_geometry_table(filename, expected_n_x, expected_x, table)
		character(len=*), intent(in) :: filename
		integer, intent(in) :: expected_n_x
		real(dp), intent(in) :: expected_x(expected_n_x)
		type(grain_geometry_table_t), intent(out) :: table
		character(len=512) :: line
		integer :: unit_id, ios_local, ix, iw, n_x_file, n_web_file, burned_value
		real(dp) :: web_max_file, x_value, web_value, area_value, p_burn_value, p_wet_value
		logical :: found_extra_data, p_burn_is_zero

		open(newunit=unit_id, file=trim(filename), status='old', action='read', iostat=ios_local)
		if (ios_local /= 0) then
			write(*, '(A)') 'Unable to open required grain geometry table: '//trim(filename)
			error stop 'Missing grain geometry table.'
		end if

		read(unit_id, '(A)', iostat=ios_local) line
		if (ios_local /= 0) then
			call grain_table_fail(unit_id, 'missing GRAIN_LEVELSET_TABLE_V1 header')
		end if
		if (trim(adjustl(line)) /= '# GRAIN_LEVELSET_TABLE_V1') then
			call grain_table_fail(unit_id, 'missing or unsupported GRAIN_LEVELSET_TABLE_V1 header')
		end if

		read(unit_id, *, iostat=ios_local) n_x_file, n_web_file, web_max_file
		if (ios_local /= 0) call grain_table_fail(unit_id, 'invalid dimension record')
		if (n_x_file /= expected_n_x) then
			write(*, '(A,I0,A,I0)') 'Table axial count is ', n_x_file, '; solver requires ', expected_n_x
			call grain_table_fail(unit_id, 'axial dimension does not match the finite-volume grid')
		end if
		if (n_web_file < 2) call grain_table_fail(unit_id, 'at least two web stations are required')
		if (.not. ieee_is_finite(web_max_file) .or. web_max_file <= 0.0_dp) then
			call grain_table_fail(unit_id, 'invalid maximum web distance')
		end if

		read(unit_id, '(A)', iostat=ios_local) line
		if (ios_local /= 0) call grain_table_fail(unit_id, 'missing column header')
		if (trim(adjustl(line)) /= '# x w A_port P_burn P_wet burned_out') then
			call grain_table_fail(unit_id, 'unexpected column header')
		end if

		table%n_x = n_x_file
		table%n_web = n_web_file
		table%web_max = web_max_file
		allocate(table%x(table%n_x), table%web(table%n_web))
		allocate(table%A_port(table%n_x, table%n_web))
		allocate(table%P_burn(table%n_x, table%n_web))
		allocate(table%P_wet(table%n_x, table%n_web))
		allocate(table%burned_out(table%n_x, table%n_web))

		do ix = 1, table%n_x
			do iw = 1, table%n_web
				read(unit_id, *, iostat=ios_local) x_value, web_value, area_value, &
					p_burn_value, p_wet_value, burned_value
				if (ios_local /= 0) then
					write(*, '(A,I0,A,I0)') 'Table ended or became invalid at axial row ', ix, &
						', web row ', iw
					call grain_table_fail(unit_id, 'data-row count does not match the declared dimensions')
				end if

				if (.not. all(ieee_is_finite([x_value, web_value, area_value, &
					p_burn_value, p_wet_value]))) then
					call grain_table_fail(unit_id, 'nonfinite numeric value')
				end if
				if (area_value <= area_floor) call grain_table_fail(unit_id, 'nonpositive port area')
				if (p_burn_value < 0.0_dp .or. p_wet_value <= 0.0_dp) then
					call grain_table_fail(unit_id, 'P_burn must be nonnegative and P_wet positive')
				end if
				if (p_burn_value > p_wet_value + geometry_table_tol * max(1.0_dp, p_wet_value)) then
					call grain_table_fail(unit_id, 'P_burn exceeds P_wet')
				end if
				if (burned_value /= 0 .and. burned_value /= 1) then
					call grain_table_fail(unit_id, 'burned_out must be 0 or 1')
				end if

				p_burn_is_zero = abs(p_burn_value) <= &
					geometry_table_tol * max(1.0_dp, p_wet_value)
				if ((burned_value == 1) .neqv. p_burn_is_zero) then
					call grain_table_fail(unit_id, 'burned_out is inconsistent with P_burn')
				end if

				if (iw == 1) then
					table%x(ix) = x_value
					if (.not. geometry_values_match(x_value, expected_x(ix))) then
						write(*, '(A,I0,A,2(ES24.14E3,1X))') 'Axial coordinate mismatch at row ', ix, &
							': table, solver=', x_value, expected_x(ix)
						call grain_table_fail(unit_id, 'axial coordinates do not match cell centers')
					end if
					if (ix > 1 .and. table%x(ix) <= table%x(ix - 1)) then
						call grain_table_fail(unit_id, 'axial coordinates are not strictly increasing')
					end if
				else if (.not. geometry_values_match(x_value, table%x(ix))) then
					call grain_table_fail(unit_id, 'x changes within an axial web block')
				end if

				if (ix == 1) then
					table%web(iw) = web_value
					if (iw == 1 .and. .not. geometry_values_match(web_value, 0.0_dp)) then
						call grain_table_fail(unit_id, 'the first web station must be zero')
					end if
					if (iw > 1 .and. table%web(iw) <= table%web(iw - 1)) then
						call grain_table_fail(unit_id, 'web coordinates are not strictly increasing')
					end if
				else if (.not. geometry_values_match(web_value, table%web(iw))) then
					call grain_table_fail(unit_id, 'web grid is inconsistent between axial blocks')
				end if

				table%A_port(ix, iw) = area_value
				table%P_burn(ix, iw) = max(p_burn_value, 0.0_dp)
				table%P_wet(ix, iw) = max(p_wet_value, 0.0_dp)
				table%burned_out(ix, iw) = burned_value
			end do
		end do

		if (.not. geometry_values_match(table%web(table%n_web), table%web_max)) then
			call grain_table_fail(unit_id, 'declared maximum web does not match the last web station')
		end if
		do ix = 1, table%n_x
			if (table%P_burn(ix, table%n_web) > &
				geometry_table_tol * max(1.0_dp, table%P_wet(ix, table%n_web))) then
				call grain_table_fail(unit_id, 'table ends before P_burn reaches zero')
			end if
			do iw = 2, table%n_web
				if (table%burned_out(ix, iw - 1) == 1 .and. &
					table%burned_out(ix, iw) == 0) then
					call grain_table_fail(unit_id, 'burned_out returns to zero at a later web station')
				end if
			end do
		end do

		found_extra_data = .false.
		do
			read(unit_id, '(A)', iostat=ios_local) line
			if (ios_local < 0) exit
			if (ios_local > 0) call grain_table_fail(unit_id, 'error after the expected final row')
			line = adjustl(line)
			if (len_trim(line) > 0 .and. line(1:1) /= '#') then
				found_extra_data = .true.
				exit
			end if
		end do
		if (found_extra_data) then
			call grain_table_fail(unit_id, 'extra data follows the declared table')
		end if
		close(unit_id)

		write(*, '(A,I0,A,I0,A,ES14.6)') 'Loaded grain geometry table: axial rows=', &
			table%n_x, ', web rows=', table%n_web, ', web_max=', table%web_max
	end subroutine load_grain_geometry_table

	integer function web_interval(table, web_value) result(iw)
		type(grain_geometry_table_t), intent(in) :: table
		real(dp), intent(in) :: web_value
		integer :: lower, upper, middle

		if (web_value <= table%web(1)) then
			iw = 1
			return
		end if
		if (web_value >= table%web(table%n_web)) then
			iw = table%n_web - 1
			return
		end if

		lower = 1
		upper = table%n_web
		do while (upper - lower > 1)
			middle = (lower + upper) / 2
			if (table%web(middle) <= web_value) then
				lower = middle
			else
				upper = middle
			end if
		end do
		iw = lower
	end function web_interval

	pure real(dp) function local_web_spacing(table, web_value) result(spacing)
		type(grain_geometry_table_t), intent(in) :: table
		real(dp), intent(in) :: web_value
		integer :: lower, upper, middle

		if (web_value <= table%web(1)) then
			lower = 1
		else if (web_value >= table%web(table%n_web)) then
			lower = table%n_web - 1
		else
			lower = 1
			upper = table%n_web
			do while (upper - lower > 1)
				middle = (lower + upper) / 2
				if (table%web(middle) <= web_value) then
					lower = middle
				else
					upper = middle
				end if
			end do
		end if
		spacing = table%web(lower + 1) - table%web(lower)
	end function local_web_spacing

	subroutine interpolate_grain_geometry(table, web_inner, area_inner, p_burn_inner, p_wet_inner)
		type(grain_geometry_table_t), intent(in) :: table
		real(dp), intent(in) :: web_inner(:)
		real(dp), intent(out) :: area_inner(:)
		real(dp), intent(out) :: p_burn_inner(:), p_wet_inner(:)
		real(dp) :: web_use, theta, denom
		integer :: ix, iw

		if (size(web_inner) /= table%n_x .or. size(area_inner) /= table%n_x .or. &
			size(p_burn_inner) /= table%n_x .or. size(p_wet_inner) /= table%n_x) then
			error stop 'Grain geometry interpolation array-size mismatch.'
		end if
		if (any(.not. ieee_is_finite(web_inner)) .or. any(web_inner < 0.0_dp)) then
			error stop 'Invalid cumulative web supplied to geometry interpolation.'
		end if

		do ix = 1, table%n_x
			web_use = min(web_inner(ix), table%web_max)
			iw = web_interval(table, web_use)
			denom = table%web(iw + 1) - table%web(iw)
			theta = (web_use - table%web(iw)) / denom
			theta = max(0.0_dp, min(1.0_dp, theta))
			area_inner(ix) = (1.0_dp - theta) * table%A_port(ix, iw) + &
				theta * table%A_port(ix, iw + 1)
			p_burn_inner(ix) = max(0.0_dp, (1.0_dp - theta) * table%P_burn(ix, iw) + &
				theta * table%P_burn(ix, iw + 1))
			p_wet_inner(ix) = max(0.0_dp, (1.0_dp - theta) * table%P_wet(ix, iw) + &
				theta * table%P_wet(ix, iw + 1))
		end do
	end subroutine interpolate_grain_geometry

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
		state%forced_ignition_reported = .false.
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
		real(dp) :: A_throat
		A_loc(1:3) = A_loc(4)
		A_throat = A2nozzle / epsnozzle
		A_loc(nx) = A_throat
		A_loc(nx-1) = (2*A_throat + A_loc(nx-4))/3
		A_loc(nx-2) = (A_throat + 2*A_loc(nx-4))/3
		A_loc(nx - 3) = A_loc(nx - 4)
		! write(*, '(A, ES14.6)') 'A_throat = ', A_loc(nx-1)
		! A_loc(nx - 3:nx) = A_loc(nx - 4)
		P_loc(1:3) = P_loc(4)
		P_loc(nx - 3:nx) = P_loc(nx - 4)
	end subroutine fill_geometry_ghosts

	subroutine fill_perimeter_ghosts(P_loc)
		real(dp), intent(inout) :: P_loc(nx)
		P_loc(1:3) = P_loc(4)
		P_loc(nx - 3:nx) = P_loc(nx - 4)
	end subroutine fill_perimeter_ghosts

	logical function conservative_state_is_admissible(Uvec, Avalue, gamma_value) result(ok)
		real(dp), intent(in) :: Uvec(3), Avalue, gamma_value
		real(dp) :: rho_value, velocity_value, pressure_value, Ause, gamma_m1

		ok = .false.
		if (.not. all(ieee_is_finite(Uvec))) return
		if (.not. ieee_is_finite(Avalue) .or. .not. ieee_is_finite(gamma_value)) return
		if (Avalue <= area_floor .or. gamma_value <= 1.0_dp) return

		Ause = max(Avalue, area_floor)
		rho_value = Uvec(1) / Ause
		if (rho_value <= rho_floor) return
		velocity_value = Uvec(2) / Uvec(1)
		gamma_m1 = gamma_value - 1.0_dp
		pressure_value = gamma_m1 * (Uvec(3) - 0.5_dp * Uvec(2) * velocity_value) / Ause
		ok = ieee_is_finite(pressure_value) .and. pressure_value > p_floor
	end function conservative_state_is_admissible

	subroutine assert_admissible_state(U, A_loc, k, location)
		real(dp), intent(in) :: U(nvar, nx), A_loc(nx), k(nx)
		character(len=*), intent(in) :: location
		real(dp) :: rho_value, velocity_value, pressure_value, Ause
		integer :: j

		do j = 1, nx
			if (.not. conservative_state_is_admissible(U(:, j), A_loc(j), k(j))) then
				Ause = max(A_loc(j), area_floor)
				rho_value = U(1, j) / Ause
				velocity_value = U(2, j) / max(U(1, j), 1.0e-30_dp)
				pressure_value = (k(j) - 1.0_dp) * &
					(U(3, j) - 0.5_dp * U(2, j) * velocity_value) / Ause
				write(*, '(A)') 'Non-admissible conservative state detected at '//trim(location)
				write(*, '(A,I0,A,ES24.14E3)') 'cell=', j, ' x=', xlist(j)
				write(*, '(A,3(ES24.14E3,1X))') 'U=', U(:, j)
				write(*, '(A,4(ES24.14E3,1X))') 'rho,u,p,A=', rho_value, velocity_value, pressure_value, A_loc(j)
				error stop 'Euler state left the admissible set.'
			end if
		end do
	end subroutine assert_admissible_state

	subroutine limit_state_to_admissible(Ubase, Ucandidate, Aface, gamma_value, Ulimited, was_limited)
		real(dp), intent(in) :: Ubase(3), Ucandidate(3), Aface, gamma_value
		real(dp), intent(out) :: Ulimited(3)
		logical, intent(out) :: was_limited
		real(dp) :: theta, theta_lo, theta_hi, theta_mid, rho_min_face
		integer :: iter

		if (.not. conservative_state_is_admissible(Ubase, Aface, gamma_value)) then
			error stop 'Invalid cell-average anchor passed to WENO positivity limiter.'
		end if

		if (conservative_state_is_admissible(Ucandidate, Aface, gamma_value)) then
			Ulimited = Ucandidate
			was_limited = .false.
			return
		end if

		theta = 1.0_dp
		rho_min_face = 1.01_dp * rho_floor * Aface
		if (.not. ieee_is_finite(Ucandidate(1)) .or. Ucandidate(1) <= rho_min_face) then
			theta = min(theta, (Ubase(1) - rho_min_face) / &
				max(Ubase(1) - Ucandidate(1), 1.0e-30_dp))
		end if
		theta = max(0.0_dp, min(1.0_dp, theta))
		Ulimited = Ubase + theta * (Ucandidate - Ubase)

		if (.not. conservative_state_is_admissible(Ulimited, Aface, gamma_value)) then
			theta_lo = 0.0_dp
			theta_hi = theta
			do iter = 1, 60
				theta_mid = 0.5_dp * (theta_lo + theta_hi)
				Ulimited = Ubase + theta_mid * (Ucandidate - Ubase)
				if (conservative_state_is_admissible(Ulimited, Aface, gamma_value)) then
					theta_lo = theta_mid
				else
					theta_hi = theta_mid
				end if
			end do
			theta = max(0.0_dp, (1.0_dp - 1.0e-12_dp) * theta_lo)
			Ulimited = Ubase + theta * (Ucandidate - Ubase)
		end if

		if (.not. conservative_state_is_admissible(Ulimited, Aface, gamma_value)) then
			Ulimited = Ubase
		end if
		was_limited = .true.
	end subroutine limit_state_to_admissible

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
			if (.not. ieee_is_finite(Ause_j) .or. Ause_j <= area_floor .or. &
				.not. ieee_is_finite(kin(j)) .or. kin(j) <= 1.0_dp) then
				error stop 'Invalid geometry or heat-capacity ratio in primitives.'
			end if
			rho_out(j) = Uin(1, j) / Ause_j
			if (.not. ieee_is_finite(rho_out(j)) .or. rho_out(j) <= rho_floor) then
				error stop 'Non-admissible density passed to primitives.'
			end if
			vel(j) = Uin(2, j) / Uin(1, j)
			Eout(j) = Uin(3, j) / Uin(1, j)
			kminus1 = kin(j) - 1.0_dp
			u2 = vel(j) * vel(j)
			pval(j) = rho_out(j) * kminus1 * (Eout(j) - 0.5_dp * u2)
			if (.not. ieee_is_finite(vel(j)) .or. .not. ieee_is_finite(Eout(j)) .or. &
				.not. ieee_is_finite(pval(j)) .or. pval(j) <= p_floor) then
				error stop 'Non-admissible pressure or energy passed to primitives.'
			end if
			a2 = kin(j) * pval(j) / rho_out(j)
			if (.not. ieee_is_finite(a2) .or. a2 <= a2_floor) then
				error stop 'Invalid sound speed passed to primitives.'
			end if
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

	subroutine chamber_stagnation(U, A_loc, k, p0_ch, T0_ch)
		real(dp), intent(in) :: U(nvar, nx), A_loc(nx), k(nx)
		real(dp), intent(out) :: p0_ch, T0_ch
		integer, parameter :: chamber_start = 4
		integer :: chamber_end, j, n_acc
		real(dp) :: Ause_j, rho_j, u_j, E_j, k_j, p_j, R_j, T_j, a_j, M_j, fac

		p0_ch = 0.0_dp
		T0_ch = 0.0_dp
		n_acc = 0
		chamber_end = max(nx - 3, chamber_start + 5)

		do j = chamber_start, chamber_end
			Ause_j = max(A_loc(j), area_floor)
			rho_j = max(U(1, j) / Ause_j, rho_floor)
			u_j = U(2, j) / max(rho_j * Ause_j, 1.0e-20_dp)
			E_j = U(3, j) / max(rho_j * Ause_j, 1.0e-20_dp)
			k_j = max(k(j), 1.0_dp + 1.0e-6_dp)
			p_j = rho_j * (k_j - 1.0_dp) * (E_j - 0.5_dp * u_j * u_j)
			p_j = max(p_j, p_floor)
			R_j = gas_constant_from_k(k_j)
			T_j = p_j / max(rho_j * R_j, 1.0e-20_dp)
			a_j = sqrt(max(k_j * p_j / rho_j, a2_floor))
			M_j = u_j / max(a_j, 1.0e-20_dp)
			fac = 1.0_dp + 0.5_dp * (k_j - 1.0_dp) * M_j * M_j

			if (ieee_is_finite(fac) .and. ieee_is_finite(T_j) .and. ieee_is_finite(p_j) .and. fac > 0.0_dp) then
				T0_ch = T0_ch + T_j * fac
				p0_ch = p0_ch + p_j * fac ** (k_j / (k_j - 1.0_dp))
				n_acc = n_acc + 1
			end if
		end do

		if (n_acc > 0) then
			T0_ch = T0_ch / real(n_acc, dp)
			p0_ch = p0_ch / real(n_acc, dp)
		else
			T0_ch = Tp0
			p0_ch = max(p0, p_floor)
		end if

		T0_ch = max(T0_ch, Tp0)
		p0_ch = max(p0_ch, p_floor)
	end subroutine chamber_stagnation

	subroutine local_outlet_stagnation(U, A_loc, k, p0_local, T0_local)
		real(dp), intent(in) :: U(nvar, nx), A_loc(nx), k(nx)
		real(dp), intent(out) :: p0_local, T0_local
		integer, parameter :: idx = nx - 3
		real(dp) :: Ause, rho_value, velocity_value, energy_value, pressure_value
		real(dp) :: sound_speed, mach_value, factor, Rvalue

		if (.not. conservative_state_is_admissible(U(:, idx), A_loc(idx), k(idx))) then
			error stop 'Invalid last interior state in outlet stagnation calculation.'
		end if
		Ause = max(A_loc(idx), area_floor)
		rho_value = U(1, idx) / Ause
		velocity_value = U(2, idx) / U(1, idx)
		energy_value = U(3, idx) / U(1, idx)
		pressure_value = rho_value * (k(idx) - 1.0_dp) * &
			(energy_value - 0.5_dp * velocity_value * velocity_value)
		Rvalue = gas_constant_from_k(k(idx))
		sound_speed = sqrt(k(idx) * pressure_value / rho_value)
		mach_value = velocity_value / sound_speed
		factor = 1.0_dp + 0.5_dp * (k(idx) - 1.0_dp) * mach_value * mach_value
		T0_local = pressure_value / (rho_value * Rvalue) * factor
		p0_local = pressure_value * factor ** (k(idx) / (k(idx) - 1.0_dp))
	end subroutine local_outlet_stagnation

	subroutine apply_supersonic_outflow(U, A_loc, k)
		real(dp), intent(inout) :: U(nvar, nx), k(nx)
		real(dp), intent(in) :: A_loc(nx)
		integer, parameter :: interior_idx = nx - 3
		real(dp) :: scale
		integer :: j

		do j = nx - 2, nx
			scale = max(A_loc(j), area_floor) / max(A_loc(interior_idx), area_floor)
			U(:, j) = scale * U(:, interior_idx)
			k(j) = k(interior_idx)
		end do
		call assert_admissible_state(U, A_loc, k, 'supersonic outlet')
	end subroutine apply_supersonic_outflow

	subroutine apply_characteristic_subsonic_outlet(U, A_loc, k, p_back)
		real(dp), intent(inout) :: U(nvar, nx), k(nx)
		real(dp), intent(in) :: A_loc(nx), p_back
		integer, parameter :: interior_idx = nx - 3
		integer, parameter :: outlet_start_idx = nx - 2
		real(dp) :: Ause_i, Ause_j, rho_i, u_i, E_i, k_i, p_i, a_i
		real(dp) :: p_b, s_i, rho_b, a_b, j_plus, u_b, E_b
		integer :: j

		Ause_i = max(A_loc(interior_idx), area_floor)
		rho_i = max(U(1, interior_idx) / Ause_i, rho_floor)
		u_i = U(2, interior_idx) / max(rho_i * Ause_i, 1.0e-20_dp)
		E_i = U(3, interior_idx) / max(rho_i * Ause_i, 1.0e-20_dp)
		k_i = max(k(interior_idx), 1.0_dp + 1.0e-6_dp)
		p_i = rho_i * (k_i - 1.0_dp) * (E_i - 0.5_dp * u_i * u_i)
		p_i = max(p_i, p_floor)
		a_i = sqrt(max(k_i * p_i / rho_i, a2_floor))

		p_b = max(p_back, p_floor)
		s_i = p_i / (rho_i ** k_i)
		rho_b = (p_b / max(s_i, 1.0e-30_dp)) ** (1.0_dp / k_i)
		rho_b = max(rho_b, rho_floor)
		a_b = sqrt(max(k_i * p_b / rho_b, a2_floor))

		j_plus = u_i + 2.0_dp * a_i / (k_i - 1.0_dp)
		u_b = max(0.0_dp, j_plus - 2.0_dp * a_b / (k_i - 1.0_dp))
		E_b = p_b / (k_i - 1.0_dp) + 0.5_dp * rho_b * u_b * u_b

		do j = outlet_start_idx, nx
			Ause_j = max(A_loc(j), area_floor)
			U(1, j) = rho_b * Ause_j
			U(2, j) = rho_b * u_b * Ause_j
			U(3, j) = E_b * Ause_j
		end do
		k(outlet_start_idx:nx) = k_i

		t0 = omp_get_wtime()
		call assert_admissible_state(U, A_loc, k, 'subsonic outlet')
		prof_validate = prof_validate + (omp_get_wtime() - t0)
		prof_call_validate = prof_call_validate + 1
	end subroutine apply_characteristic_subsonic_outlet

	subroutine apply_choked_nozzle_outlet(U, A_loc, k)
		real(dp), intent(inout) :: U(nvar, nx), k(nx)
		real(dp), intent(in) :: A_loc(nx)
		integer, parameter :: outlet_start_idx = nx - 2
		real(dp) :: p0_ch, T0_ch, k_noz, R_noz, Athroat, m_dot
		real(dp) :: area_ratio, M_j, T_j, p_j, rho_j, u_j, E_j, Ause_j
		integer :: j

		call local_outlet_stagnation(U, A_loc, k, p0_ch, T0_ch)
		k_noz = max(k(nx - 3), 1.0_dp + 1.0e-6_dp)
		R_noz = gas_constant_from_k(k_noz)
		Athroat = A2nozzle / epsnozzle

		m_dot = Athroat * p0_ch * sqrt(k_noz / max(R_noz * T0_ch, 1.0e-20_dp)) * &
					(2.0_dp / (k_noz + 1.0_dp)) ** ((k_noz + 1.0_dp) / (2.0_dp * (k_noz - 1.0_dp)))

		do j = outlet_start_idx, nx
			Ause_j = max(A_loc(j), area_floor)
			area_ratio = Ause_j / max(Athroat, area_floor)
			M_j = solve_area_mach(area_ratio, k_noz, .false.)
			T_j = T0_ch / (1.0_dp + 0.5_dp * (k_noz - 1.0_dp) * M_j * M_j)
			p_j = p0_ch / (1.0_dp + 0.5_dp * (k_noz - 1.0_dp) * M_j * M_j) ** (k_noz / (k_noz - 1.0_dp))
			rho_j = p_j / max(R_noz * T_j, 1.0e-20_dp)
			u_j = m_dot / max(rho_j * Ause_j, 1.0e-20_dp)
			E_j = p_j / (k_noz - 1.0_dp) + 0.5_dp * rho_j * u_j * u_j

			U(1, j) = rho_j * Ause_j
			U(2, j) = rho_j * u_j * Ause_j
			U(3, j) = E_j * Ause_j
		end do

		k(outlet_start_idx:nx) = k_noz

		t0 = omp_get_wtime()
		call assert_admissible_state(U, A_loc, k, 'choked outlet')
		prof_validate = prof_validate + (omp_get_wtime() - t0)
		prof_call_validate = prof_call_validate + 1
	end subroutine apply_choked_nozzle_outlet

	subroutine apply_nozzle_outlet(U, A_loc, k)
		real(dp), intent(inout) :: U(nvar, nx), k(nx)
		real(dp), intent(in) :: A_loc(nx)
		integer, parameter :: fictitious_cell_idx = nx - 3
		real(dp) :: rho_f, u_f, E_f, k_f, p_f, p0_ch, T0_ch, a_f, M_f
		real(dp) :: critical_ratio

		rho_f = max(U(1, fictitious_cell_idx) / max(A_loc(fictitious_cell_idx), area_floor), rho_floor)
		u_f = U(2, fictitious_cell_idx) / max(rho_f * A_loc(fictitious_cell_idx), 1.0e-20_dp)
		E_f = U(3, fictitious_cell_idx) / max(rho_f * A_loc(fictitious_cell_idx), 1.0e-20_dp)
		k_f = max(k(fictitious_cell_idx), 1.0_dp + 1.0e-6_dp)
		p_f = rho_f * max(k_f - 1.0_dp, 1.0e-8_dp) * (E_f - 0.5_dp * u_f * u_f)
		a_f = sqrt(max(k_f * max(p_f, p_floor) / rho_f, a2_floor))
		M_f = u_f / a_f

		if (.not.(ieee_is_finite(rho_f) .and. ieee_is_finite(u_f) .and. ieee_is_finite(p_f))) then
			call apply_characteristic_subsonic_outlet(U, A_loc, k, p0)
			return
		end if
		if (rho_f <= 1.0e-12_dp .or. p_f <= max(1.0_dp, p0) .or. u_f <= 0.0_dp) then
			call apply_characteristic_subsonic_outlet(U, A_loc, k, p0)
			return
		end if
		if (M_f >= 1.0_dp) then
			call apply_supersonic_outflow(U, A_loc, k)
			return
		end if

		call local_outlet_stagnation(U, A_loc, k, p0_ch, T0_ch)

		critical_ratio = (2.0_dp / (k_f + 1.0_dp)) ** (k_f / (k_f - 1.0_dp))
		if (p0 / p0_ch >= critical_ratio) then
			call apply_characteristic_subsonic_outlet(U, A_loc, k, p0)
			return
		end if

		call apply_choked_nozzle_outlet(U, A_loc, k)
	end subroutine apply_nozzle_outlet

	subroutine supersonic_properties(U, A_loc, k, M_exit, T_exit, p_exit, rho_exit, u_exit, E_exit, thrust)
		real(dp), intent(inout) :: U(nvar, nx), k(nx)
		real(dp), intent(in) :: A_loc(nx)
		integer, parameter :: nozzle_state_idx = nx - 2
		real(dp) :: k_s, R_s, p0_ch, T0_ch, critical_ratio, m_dot
		real(dp) :: M_exit, T_exit, p_exit, rho_exit, u_exit, E_exit, thrust

		k_s = max(k(nozzle_state_idx), 1.0_dp + 1.0e-6_dp)
		R_s = gas_constant_from_k(k_s)
		call local_outlet_stagnation(U, A_loc, k, p0_ch, T0_ch)

		critical_ratio = (2.0_dp / (k_s + 1.0_dp)) ** (k_s / (k_s - 1.0_dp))
		if (p0 / p0_ch >= critical_ratio) then
			M_exit = 0.0_dp
			T_exit = 0.0_dp
			p_exit = 0.0_dp
			rho_exit = 0.0_dp
			u_exit = 0.0_dp
			E_exit = 0.0_dp
			thrust = 0.0_dp
			return
		end if

		write(*,'(A,4(ES14.6,1X))') &
    		'exit diagnostic: t,p0_ch,p0/p0_ch,critical=', &
    		t, p0_ch, p0/p0_ch, critical_ratio

		M_exit = solve_area_mach(epsnozzle, k_s, .true.)
		T_exit = T0_ch / (1.0_dp + 0.5_dp * (k_s - 1.0_dp) * M_exit * M_exit)
		p_exit = p0_ch / (1.0_dp + 0.5_dp * (k_s - 1.0_dp) * M_exit * M_exit) ** (k_s / (k_s - 1.0_dp))
		rho_exit = p_exit / max(R_s * T_exit, 1.0e-20_dp)

		m_dot = (A2nozzle / epsnozzle) * p0_ch * sqrt(k_s / max(R_s * T0_ch, 1.0e-20_dp)) * &
					(2.0_dp / (k_s + 1.0_dp)) ** ((k_s + 1.0_dp) / (2.0_dp * (k_s - 1.0_dp)))
		u_exit = m_dot / max(rho_exit * A2nozzle, 1.0e-20_dp)
		E_exit = p_exit / (k_s - 1.0_dp) + 0.5_dp * rho_exit * u_exit * u_exit
		thrust = max(m_dot * u_exit + (p_exit - p0) * A2nozzle, 0.0_dp)
		! write(*, '(A, ES14.6)') 'M_exit = ', M_exit
	end subroutine supersonic_properties


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
		k(3) = k(4)
		k(2) = k(5)
		k(1) = k(6)

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

	subroutine weno5_reconstruct(U, A_loc, k, UL, UR, eigenvals, khat_out, limited_iface)
		real(dp), intent(in) :: U(nvar, nx), A_loc(nx), k(nx)
		real(dp), intent(out) :: UL(nvar, nx - 5), UR(nvar, nx - 5), eigenvals(nvar, nx - 5), khat_out(nx - 5)
		logical, intent(out) :: limited_iface(nx - 5)
		integer :: n_ifaces, n_limited
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
		real(dp) :: A_im1, A_i, Aface, kLm1, kRm1
		real(dp) :: UbaseL(3), UbaseR(3), UcandidateL(3), UcandidateR(3)
		logical :: limitedL, limitedR
		integer :: r

		eps_weno = 1.0e-6_dp
		n_ifaces = nx - 5
		n_limited = 0
		limited_iface = .false.

		!$omp parallel do default(shared) &
		!$omp   shared(U, A_loc, k, UL, UR, eigenvals, khat_out, limited_iface, &
		!$omp          n_ifaces, eps_weno, rho_floor, p_floor) &
		!$omp   private(j, r, uim3, uim2, uim1, ui, uip1, uip2, &
		!$omp           rhoL, velL, EL, kL, pL, HL, rhoR, velR, ER, kR, pR, HR, &
		!$omp           sL, sR, denom, rhohat, uhat, Hhat, khat, ahat2, ahat, &
		!$omp           Pmat, Lmat, wuL, wuR, wm, &
		!$omp           p0L, p1L, p2L, p0R, p1R, p2R, &
		!$omp           dw1, dw2, dw3, b1, b2, b3, a1, a2, a3, asum, &
		!$omp           A_im1, A_i, Aface, kLm1, kRm1, UbaseL, UbaseR, &
		!$omp           UcandidateL, UcandidateR, limitedL, limitedR) &
		!$omp   reduction(+:n_limited) &
		!$omp   schedule(static)
		do j = 1, n_ifaces
			uim3 = U(:, j) / max(A_loc(j), area_floor)
			uim2 = U(:, j + 1) / max(A_loc(j + 1), area_floor)
			uim1 = U(:, j + 2) / max(A_loc(j + 2), area_floor)
			ui = U(:, j + 3) / max(A_loc(j + 3), area_floor)
			uip1 = U(:, j + 4) / max(A_loc(j + 4), area_floor)
			uip2 = U(:, j + 5) / max(A_loc(j + 5), area_floor)

			A_im1 = max(A_loc(j + 2), area_floor)
			A_i = max(A_loc(j + 3), area_floor)
			Aface = max(0.5_dp * (A_im1 + A_i), area_floor)

			rhoL = max(uim1(1), rho_floor)
			velL = uim1(2) / max(rhoL, 1.0e-20_dp)
			EL = uim1(3) / max(rhoL, 1.0e-20_dp)
			kL = k(j + 2)
			kLm1 = max(kL - 1.0_dp, 1.0e-8_dp)
			pL = max(rhoL * kLm1 * (EL - 0.5_dp * velL * velL), p_floor)
			HL = EL + pL / rhoL

			rhoR = max(ui(1), rho_floor)
			velR = ui(2) / max(rhoR, 1.0e-20_dp)
			ER = ui(3) / max(rhoR, 1.0e-20_dp)
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

			UcandidateL = Aface * matmul(Pmat, wuL)
			UcandidateR = Aface * matmul(Pmat, wuR)
			UbaseL = (Aface / A_im1) * U(:, j + 2)
			UbaseR = (Aface / A_i) * U(:, j + 3)
			call limit_state_to_admissible(UbaseL, UcandidateL, Aface, kL, UL(:, j), limitedL)
			call limit_state_to_admissible(UbaseR, UcandidateR, Aface, kR, UR(:, j), limitedR)
			limited_iface(j) = limitedL .or. limitedR
			if (limitedL) n_limited = n_limited + 1
			if (limitedR) n_limited = n_limited + 1
			eigenvals(:, j) = [uhat, uhat + ahat, uhat - ahat]
			khat_out(j) = khat
		end do
		!$omp end parallel do
		reconstruction_limit_count = reconstruction_limit_count + n_limited
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
		logical :: limited_iface(nx - 5), troubled(nx - 5)
		integer :: j

		t0 = omp_get_wtime()
		call weno5_reconstruct(U, A_loc, k, UL, UR, eig, khat, limited_iface)
		prof_weno = prof_weno + (omp_get_wtime() - t0)
		prof_call_weno = prof_call_weno + 1
		call primitives(UL, A_loc, khat, rhoL_arr, velL_arr, pL_arr, eL_arr, aL_arr)
		call primitives(UR, A_loc, khat, rhoR_arr, velR_arr, pR_arr, eR_arr, aR_arr)
		call detect_troubled_from_recon(UL, UR, A_loc, k, limited_iface, troubled)

		n_ifaces = nx - 5
		maxabs = 0.0_dp

		do j = 1, n_ifaces
			if (troubled(j)) then
				call cell_average_interface_speeds(U, A_loc, k, j, SL(j), SR(j))
				maxabs = max(maxabs, abs(SL(j)), abs(SR(j)))
				cycle
			end if
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


	subroutine euler_flux_from_prims(rho_in, vel_in, pval_in, Eout_in, A_loc, nloc, flux)
		integer, intent(in) :: nloc
		real(dp), intent(in) :: rho_in(nloc), vel_in(nloc), pval_in(nloc), Eout_in(nloc), A_loc(nx)
		real(dp), intent(out) :: flux(3, nloc)
		real(dp) :: Ause_j
		integer :: j
		do j = 1, nloc
			Ause_j = max(0.5_dp * (A_loc(j + 2) + A_loc(j + 3)), area_floor)
			flux(1, j) = rho_in(j) * vel_in(j) * Ause_j
			flux(2, j) = (rho_in(j) * vel_in(j) * vel_in(j) + pval_in(j)) * Ause_j
			flux(3, j) = vel_in(j) * (rho_in(j) * Eout_in(j) + pval_in(j)) * Ause_j
		end do
	end subroutine euler_flux_from_prims

	subroutine wave_speeds_from_prims(rhoL_arr, velL_arr, pL_arr, aL_arr, &
			rhoR_arr, velR_arr, pR_arr, aR_arr, khat, n_ifaces, SL, SR, maxabs)
		integer, intent(in) :: n_ifaces
		real(dp), intent(in) :: rhoL_arr(n_ifaces), velL_arr(n_ifaces), pL_arr(n_ifaces), aL_arr(n_ifaces)
		real(dp), intent(in) :: rhoR_arr(n_ifaces), velR_arr(n_ifaces), pR_arr(n_ifaces), aR_arr(n_ifaces)
		real(dp), intent(in) :: khat(n_ifaces)
		real(dp), intent(out) :: SL(n_ifaces), SR(n_ifaces), maxabs
		real(dp) :: khat_safe, kh_m1, kh_p1, inv2k, gexp, pexp, pL_gamma, pR_gamma
		real(dp) :: base_num, base_den, base, pstarr, coeff, qL, qR, pL_j, pR_j
		integer :: j

		maxabs = 0.0_dp
		do j = 1, n_ifaces
			pL_j = max(pL_arr(j), p_floor)
			pR_j = max(pR_arr(j), p_floor)
			khat_safe = max(khat(j), 1.0_dp + 1.0e-6_dp)
			kh_m1 = khat_safe - 1.0_dp
			kh_p1 = khat_safe + 1.0_dp
			inv2k = 1.0_dp / (2.0_dp * khat_safe)
			gexp = kh_m1 * inv2k
			pexp = 2.0_dp * khat_safe / kh_m1
			pL_gamma = pL_j ** gexp
			pR_gamma = pR_j ** gexp
			base_num = aL_arr(j) + aR_arr(j) - 0.5_dp * kh_m1 * (velR_arr(j) - velL_arr(j))
			base_den = aL_arr(j) / pL_gamma + aR_arr(j) / pR_gamma
			base = max(base_num / regularize_denom(base_den), 1.0e-16_dp)
			pstarr = max(base ** pexp, p_floor)
			coeff = kh_p1 * inv2k
			if (pstarr <= pL_j) then
				qL = 1.0_dp
			else
				qL = sqrt(max(1.0_dp + coeff * (pstarr / pL_j - 1.0_dp), 1.0_dp))
			end if
			if (pstarr <= pR_j) then
				qR = 1.0_dp
			else
				qR = sqrt(max(1.0_dp + coeff * (pstarr / pR_j - 1.0_dp), 1.0_dp))
			end if
			SL(j) = velL_arr(j) - aL_arr(j) * qL
			SR(j) = velR_arr(j) + aR_arr(j) * qR
			maxabs = max(maxabs, abs(SL(j)), abs(SR(j)))
		end do
	end subroutine wave_speeds_from_prims

	subroutine detect_troubled_from_recon(UL, UR, A_loc, k, limited_iface, troubled)
		real(dp), intent(in) :: UL(3, nx - 5), UR(3, nx - 5), A_loc(nx), k(nx)
		logical, intent(in) :: limited_iface(nx - 5)
		logical, intent(out) :: troubled(nx - 5)
		integer :: n_ifaces, j
		real(dp) :: Ause_l, Ause_r, rhoL_u, rhoR_u, rhoL_val, rhoR_val
		real(dp) :: velL2, velR2, EL, ER, kLm1, kRm1, pL_u, pR_u, psum, jump

		n_ifaces = nx - 5
		! Positivity limiting means that the unlimited WENO state was not
		! admissible.  Keep that interface on the robust cell-average path even
		! when the limited state lies just inside the admissible set.
		troubled = limited_iface
		do j = 1, n_ifaces
			Ause_l = max(0.5_dp * (A_loc(j + 2) + A_loc(j + 3)), area_floor)
			Ause_r = Ause_l
			rhoL_u = UL(1, j) / Ause_l
			rhoR_u = UR(1, j) / Ause_r
			if (rhoL_u <= reconstruction_floor_factor * rho_floor .or. &
				rhoR_u <= reconstruction_floor_factor * rho_floor) troubled(j) = .true.
			rhoL_val = max(rhoL_u, rho_floor); rhoR_val = max(rhoR_u, rho_floor)
			velL2 = (UL(2, j) / max(rhoL_val * Ause_l, 1.0e-20_dp)) ** 2
			velR2 = (UR(2, j) / max(rhoR_val * Ause_r, 1.0e-20_dp)) ** 2
			EL = UL(3, j) / max(rhoL_val * Ause_l, 1.0e-20_dp)
			ER = UR(3, j) / max(rhoR_val * Ause_r, 1.0e-20_dp)
			kLm1 = max(k(j + 2) - 1.0_dp, 1.0e-8_dp); kRm1 = max(k(j + 3) - 1.0_dp, 1.0e-8_dp)
			pL_u = rhoL_val * kLm1 * (EL - 0.5_dp * velL2)
			pR_u = rhoR_val * kRm1 * (ER - 0.5_dp * velR2)
			if (pL_u <= reconstruction_floor_factor * p_floor .or. &
				pR_u <= reconstruction_floor_factor * p_floor) troubled(j) = .true.
			psum = max(0.5_dp * (max(pL_u, p_floor) + max(pR_u, p_floor)), p_floor)
			jump = abs(max(pL_u, p_floor) - max(pR_u, p_floor)) / psum
			if (jump > 0.5_dp) troubled(j) = .true.
		end do
	end subroutine detect_troubled_from_recon


	subroutine detect_troubled_interfaces(U, A_loc, k, troubled)
		real(dp), intent(in) :: U(nvar, nx), A_loc(nx), k(nx)
		logical, intent(out) :: troubled(nx - 5)
		real(dp) :: UL(3, nx - 5), UR(3, nx - 5), eig(3, nx - 5), khat(nx - 5)
		logical :: limited_iface(nx - 5)

		t0 = omp_get_wtime()
		call weno5_reconstruct(U, A_loc, k, UL, UR, eig, khat, limited_iface)
		prof_weno = prof_weno + (omp_get_wtime() - t0)
		prof_call_weno = prof_call_weno + 1
		call detect_troubled_from_recon(UL, UR, A_loc, k, limited_iface, troubled)
	end subroutine detect_troubled_interfaces

	subroutine cell_average_interface_speeds(U, A_loc, k, iface, SL_cell, SR_cell)
		real(dp), intent(in) :: U(nvar, nx), A_loc(nx), k(nx)
		integer, intent(in) :: iface
		real(dp), intent(out) :: SL_cell, SR_cell
		integer :: iL, iR
		real(dp) :: AL, AR, rhoL, rhoR, uL, uR, pL, pR, aLL, aRR

		iL = iface + 2
		iR = iface + 3
		AL = max(A_loc(iL), area_floor)
		AR = max(A_loc(iR), area_floor)
		if (.not. conservative_state_is_admissible(U(:, iL), AL, k(iL)) .or. &
			.not. conservative_state_is_admissible(U(:, iR), AR, k(iR))) then
			error stop 'Invalid cell average supplied to CFL wave-speed fallback.'
		end if

		rhoL = U(1, iL) / AL
		rhoR = U(1, iR) / AR
		uL = U(2, iL) / U(1, iL)
		uR = U(2, iR) / U(1, iR)
		pL = (k(iL) - 1.0_dp) * (U(3, iL) - 0.5_dp * U(2, iL) * uL) / AL
		pR = (k(iR) - 1.0_dp) * (U(3, iR) - 0.5_dp * U(2, iR) * uR) / AR
		aLL = sqrt(k(iL) * pL / rhoL)
		aRR = sqrt(k(iR) * pR / rhoR)

		SL_cell = min(uL - aLL, uR - aRR)
		SR_cell = max(uL + aLL, uR + aRR)
	end subroutine cell_average_interface_speeds

	subroutine first_order_hlle_interface(U, A_loc, k, iface, flux)
		real(dp), intent(in) :: U(nvar, nx), A_loc(nx), k(nx)
		integer, intent(in) :: iface
		real(dp), intent(out) :: flux(3)
		integer :: iL, iR
		real(dp) :: AL, AR, Aface, rhoL, rhoR, uL, uR, EL, ER, pL, pR, aLL, aRR
		real(dp) :: SL, SR, denom, ULface(3), URface(3), fL(3), fR(3)

		iL = iface + 2
		iR = iface + 3
		AL = max(A_loc(iL), area_floor)
		AR = max(A_loc(iR), area_floor)
		Aface = max(0.5_dp * (AL + AR), area_floor)
		if (.not. conservative_state_is_admissible(U(:, iL), AL, k(iL)) .or. &
			.not. conservative_state_is_admissible(U(:, iR), AR, k(iR))) then
			error stop 'Invalid cell average supplied to first-order HLLE flux.'
		end if

		rhoL = U(1, iL) / AL
		rhoR = U(1, iR) / AR
		uL = U(2, iL) / U(1, iL)
		uR = U(2, iR) / U(1, iR)
		EL = U(3, iL) / U(1, iL)
		ER = U(3, iR) / U(1, iR)
		pL = (k(iL) - 1.0_dp) * (U(3, iL) - 0.5_dp * U(2, iL) * uL) / AL
		pR = (k(iR) - 1.0_dp) * (U(3, iR) - 0.5_dp * U(2, iR) * uR) / AR
		aLL = sqrt(k(iL) * pL / rhoL)
		aRR = sqrt(k(iR) * pR / rhoR)

		ULface = (Aface / AL) * U(:, iL)
		URface = (Aface / AR) * U(:, iR)
		fL = [rhoL * uL * Aface, (rhoL * uL * uL + pL) * Aface, &
			uL * (rhoL * EL + pL) * Aface]
		fR = [rhoR * uR * Aface, (rhoR * uR * uR + pR) * Aface, &
			uR * (rhoR * ER + pR) * Aface]
		SL = min(uL - aLL, uR - aRR)
		SR = max(uL + aLL, uR + aRR)

		if (SL >= 0.0_dp) then
			flux = fL
		else if (SR <= 0.0_dp) then
			flux = fR
		else
			denom = SR - SL
			if (denom <= eps_denom) error stop 'Degenerate HLLE wave-speed interval.'
			flux = (SR * fL - SL * fR + SL * SR * (URface - ULface)) / denom
		end if
	end subroutine first_order_hlle_interface

	subroutine hlle_flux(U, A_loc, k, fm, fp)
		real(dp), intent(in) :: U(nvar, nx), A_loc(nx), k(nx)
		real(dp), intent(out) :: fm(3, nx - 6), fp(3, nx - 6)
		integer :: n_ifaces
		real(dp) :: fhlle(3, nx - 5)
		integer :: j

		n_ifaces = nx - 5
		do j = 1, n_ifaces
			call first_order_hlle_interface(U, A_loc, k, j, fhlle(:, j))
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
		real(dp) :: Ause, denom_sstar, denom_l, denom_r, denom_e_l, denom_e_r
		real(dp) :: fhlle(3, nx - 5), fblend(3, nx - 5), theta(nx - 5)
		logical :: limited_iface(nx - 5), troubled(nx - 5)
		logical :: need_hlle
		integer :: j

		n_ifaces = nx - 5

		! *** SINGLE WENO5 call — used for everything ***
		t0 = omp_get_wtime()
		call weno5_reconstruct(U, A_loc, k, UL, UR, eig, khat, limited_iface)
		prof_weno = prof_weno + (omp_get_wtime() - t0)
		prof_call_weno = prof_call_weno + 1


		! Primitives from reconstructed states (computed once)
		call primitives(UL, A_loc, khat, rhoL_arr, velL_arr, pL_arr, eL_arr, aL_arr)
		call primitives(UR, A_loc, khat, rhoR_arr, velR_arr, pR_arr, eR_arr, aR_arr)

		! Euler fluxes from pre-computed primitives (no redundant primitives call)
		call euler_flux_from_prims(rhoL_arr, velL_arr, pL_arr, eL_arr, A_loc, n_ifaces, fL)
		call euler_flux_from_prims(rhoR_arr, velR_arr, pR_arr, eR_arr, A_loc, n_ifaces, fR)

		! Wave speeds from pre-computed primitives (no redundant WENO5 + primitives)
		call wave_speeds_from_prims(rhoL_arr, velL_arr, pL_arr, aL_arr, &
			rhoR_arr, velR_arr, pR_arr, aR_arr, khat, n_ifaces, SL, SR, maxabs)
		call detect_troubled_from_recon(UL, UR, A_loc, k, limited_iface, troubled)

		! HLLC star-state computation
		do j = 1, n_ifaces
			if (troubled(j)) then
				call first_order_hlle_interface(U, A_loc, k, j, f(:, j))
				cycle
			end if
			Ause = max(0.5_dp * (A_loc(j + 2) + A_loc(j + 3)), area_floor)
			denom_sstar = rhoL_arr(j) * (SL(j) - velL_arr(j)) - rhoR_arr(j) * (SR(j) - velR_arr(j))
			if (.not. ieee_is_finite(denom_sstar) .or. abs(denom_sstar) <= eps_denom) then
				troubled(j) = .true.
				call first_order_hlle_interface(U, A_loc, k, j, f(:, j))
				cycle
			end if
			Sstar(j) = (pR_arr(j) - pL_arr(j) + &
				rhoL_arr(j) * velL_arr(j) * (SL(j) - velL_arr(j)) - &
				rhoR_arr(j) * velR_arr(j) * (SR(j) - velR_arr(j))) / denom_sstar
			denom_l = SL(j) - Sstar(j)
			denom_r = SR(j) - Sstar(j)
			denom_e_l = rhoL_arr(j) * (SL(j) - velL_arr(j))
			denom_e_r = rhoR_arr(j) * (SR(j) - velR_arr(j))
			if (.not. ieee_is_finite(Sstar(j)) .or. abs(denom_l) <= eps_denom .or. &
				abs(denom_r) <= eps_denom .or. abs(denom_e_l) <= eps_denom .or. &
				abs(denom_e_r) <= eps_denom) then
				troubled(j) = .true.
				call first_order_hlle_interface(U, A_loc, k, j, f(:, j))
				cycle
			end if
			UstarL(1, j) = Ause * rhoL_arr(j) * (SL(j) - velL_arr(j)) / denom_l
			UstarL(2, j) = UstarL(1, j) * Sstar(j)
			UstarL(3, j) = Ause * rhoL_arr(j) * (SL(j) - velL_arr(j)) / denom_l * &
				(eL_arr(j) + (Sstar(j) - velL_arr(j)) * &
				(Sstar(j) + pL_arr(j) / denom_e_l))
			UstarR(1, j) = Ause * rhoR_arr(j) * (SR(j) - velR_arr(j)) / denom_r
			UstarR(2, j) = UstarR(1, j) * Sstar(j)
			UstarR(3, j) = Ause * rhoR_arr(j) * (SR(j) - velR_arr(j)) / denom_r * &
				(eR_arr(j) + (Sstar(j) - velR_arr(j)) * &
				(Sstar(j) + pR_arr(j) / denom_e_r))
			if (.not. conservative_state_is_admissible(UstarL(:, j), Ause, khat(j)) .or. &
				.not. conservative_state_is_admissible(UstarR(:, j), Ause, khat(j))) then
				troubled(j) = .true.
				call first_order_hlle_interface(U, A_loc, k, j, f(:, j))
				cycle
			end if
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

		fm = f(:, 1:nx - 6); fp = f(:, 2:nx - 5)

		need_hlle = any(troubled)
		if (.not. need_hlle) return

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
					call first_order_hlle_interface(U, A_loc, k, j, fhlle(:, j))
					fblend(:, j) = theta(j) * f(:, j) + (1.0_dp - theta(j)) * fhlle(:, j)
			else
				fblend(:, j) = f(:, j)
			end if
		end do
		fm = fblend(:, 1:nx - 6); fp = fblend(:, 2:nx - 5)
	end subroutine hllc_flux


		subroutine erosive_burning(U, A_loc, Pwet_loc, k, erosive_on, rb, G)
			real(dp), intent(in) :: U(nvar, nx), A_loc(nx), Pwet_loc(nx), k(nx)
			logical, intent(in) :: erosive_on
		real(dp), intent(out) :: rb(nx - 7), G(nx - 7)
		real(dp) :: rho_arr(nx), vel_arr(nx), pval_arr(nx), Eout_arr(nx), a_arr(nx)
		real(dp) :: Dh, rb0, r_iter, expo, re_scale, r_new, G_loc, G_use
		integer :: j, it

		t0 = omp_get_wtime()
		call primitives(U, A_loc, k, rho_arr, vel_arr, pval_arr, Eout_arr, a_arr)
		prof_primitives = prof_primitives + (omp_get_wtime() - t0)
		prof_call_primitives = prof_call_primitives + 1


		rb = 0.0_dp
		G = 0.0_dp
		do j = 1, nx - 7
			G_loc = rho_arr(j + 3) * vel_arr(j + 3)
			G_use = max(G_loc, 0.0_dp)
			! Hydraulic diameter is based on the entire gas-wetted perimeter,
			! including casing exposed after partial propellant contact.
			Dh = 4.0_dp * A_loc(j + 3) / max(Pwet_loc(j + 3), 1.0e-20_dp)
			rb0 = arb * (pval_arr(j + 3) ** nrb)
			r_new = rb0
			G(j) = G_loc

			if (erosive_on .and. G_use > mass_flux_floor .and. Dh > area_floor) then
				re_scale = (G_use ** 0.8_dp) * (Dh ** (-0.2_dp))
				r_iter = rb0
				do it = 1, 24
					expo = -beta_er * rhosolid * r_iter / G_use
					expo = max(-60.0_dp, min(0.0_dp, expo))
					r_new = rb0 + alpha_er * re_scale * exp(expo)
					if (.not. ieee_is_finite(r_new)) then
						write(*, '(A,I0,A,4(ES24.14E3,1X))') 'Invalid erosive rate at cell ', j + 3, &
							': G,Dh,rb0,r_iter=', G_loc, Dh, rb0, r_iter
						error stop 'Nonfinite Lenoir-Robillard iteration.'
					end if
					if (abs(r_new - r_iter) <= 1.0e-8_dp * max(rb0, r_new)) exit
					r_iter = r_new
				end do
			end if
			rb(j) = max(rb0, r_new)
		end do
	end subroutine erosive_burning

	subroutine update_ignition_thermal_state(U, A_loc, Pwet_loc, state, dt_local, k)
		real(dp), intent(in) :: U(nvar, nx), A_loc(nx), Pwet_loc(nx), dt_local, k(nx)
		type(thermal_state_t), intent(inout) :: state
		real(dp) :: rho(nx), vel(nx), pval(nx), Eout(nx), a(nx)
		real(dp) :: rho_i, p_i, Rloc, muloc, Kloc, cploc
		real(dp) :: Tg, Dh, Re, Pr, Nu, htc, alpha_s, coeff
		real(dp) :: t_old, t_new, kernel, sqrt_dt, acoef
		real(dp) :: history_term(size(state%Ts)), Ts_new(size(state%Ts)), q_new(size(state%Ts))
		integer :: j, h

		call primitives(U, A_loc, k, rho, vel, pval, Eout, a)
		history_term = 0.0_dp

		t_old = state%time
		t_new = t_old + dt_local

		if (state%n_hist > 0) then
			do h = 1, state%n_hist
				kernel = sqrt(max(t_new - state%t_hist(h), 0.0_dp)) - sqrt(max(t_new - state%t_hist(h + 1), 0.0_dp))
				history_term = history_term + state%q_hist(:, h) * kernel
			end do
		end if

		alpha_s = Ksolid / (rhosolid * cpsolid)
		coeff = 2.0_dp * sqrt(alpha_s) / (Ksolid * sqrt(acos(-1.0_dp)))
		sqrt_dt = sqrt(max(dt_local, 1.0e-12_dp))
		do j = 1, size(state%Ts)
			rho_i = rho(j + 3)
			p_i = pval(j + 3)
			if (use_single_gas_eos) then
				Rloc = Rgas
				muloc = mugas
				Kloc = Kgas
				cploc = cpgas
			else if (state%ignited(j)) then
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
			! Heat transfer also sees the full gas-wetted perimeter.
			Dh = 4.0_dp * A_loc(j + 3) / max(Pwet_loc(j + 3), 1.0e-20_dp)
			
			! Re = max(Re, 1.0_dp + 1.0e-12_dp)
			! Pr = max(Pr, 1.0e-12_dp)
			! if (Re >= 3000.0_dp) then
			! 	f = (0.79_dp * log(Re) - 1.64_dp) ** (-2.0_dp)
			! 	Nu = ((f / 8.0_dp) * (Re - 1000.0_dp) * Pr) / &
			! 		(1.0_dp + 12.7_dp * sqrt(f / 8.0_dp) * (Pr ** (2.0_dp / 3.0_dp) - 1.0_dp))
			Re = rho_i * abs(vel(j + 3)) * Dh / max(muloc, 1.0e-20_dp)
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

			if (j < 10) then
				! write Tg for j < 10 to check igniter gas properties
				! write(*, '(A, I0, A, ES14.6)') 'j=', j, ' Tg=', Tg
			end if
		end do

		!write maximum and minimum Ts for debugging
		! write(*, '(A, ES14.6)') 'max Ts=', maxval(Ts_new)
		! write(*, '(A, ES14.6)') 'min Ts=', minval(Ts_new)

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

	subroutine report_and_force_ignition(state, force_time)
		type(thermal_state_t), intent(inout) :: state
		real(dp), intent(in) :: force_time
		integer :: j, n_unignited

		if (state%forced_ignition_reported) then
			state%ignited = .true.
			return
		end if

		n_unignited = count(.not. state%ignited)
		write(*, '(A,ES14.6,A,I0)') 'Forced ignition at t=', force_time, &
			'; cells still naturally unignited=', n_unignited
		if (n_unignited > 0) then
			write(*, '(A)') 'cell_index  x  surface_temperature'
			do j = 1, size(state%ignited)
				if (.not. state%ignited(j)) then
					write(*, '(I0,1X,2(ES24.14E3,1X))') j + 3, xlist(j + 3), state%Ts(j)
				end if
			end do
		end if

		state%ignited = .true.
		state%forced_ignition_reported = .true.
	end subroutine report_and_force_ignition

	subroutine source_term(U, A_loc, Pburn_loc, Pwet_loc, ignited, dx_local, t_local, &
		k, erosive_on, S, rb, G)
		real(dp), intent(in) :: U(nvar, nx), A_loc(nx), Pburn_loc(nx), Pwet_loc(nx)
		real(dp), intent(in) :: dx_local, t_local, k(nx)
		logical, intent(in) :: ignited(nx - 7), erosive_on
		real(dp), intent(out) :: S(3, nx - 6), rb(nx - 7), G(nx - 7)
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
		G = 0.0_dp
		t0 = omp_get_wtime()
		call erosive_burning(U, A_loc, Pwet_loc, k, erosive_on, rb, G)
		prof_erosive = prof_erosive + (omp_get_wtime() - t0)
		prof_call_erosive = prof_call_erosive + 1
		do j = 1, nx - 7
			if (.not. ignited(j)) then
				rb(j) = 0.0_dp
			else if (Pburn_loc(j + 3) <= 0.0_dp) then
				! Casing contact does not stop regression while any propellant
				! perimeter remains.  Only P_burn == 0 terminates local burning.
				rb(j) = 0.0_dp
			else if (.not. ieee_is_finite(rb(j)) .or. rb(j) <= 0.0_dp) then
				write(*, '(A,I0,A,2(ES24.14E3,1X))') 'Invalid burn rate in ignited cell ', &
					j + 3, ': rb,G=', rb(j), G(j)
				error stop 'Ignited cell lost its positive burning rate.'
			end if
			S(1, j) = rb(j) * Pburn_loc(j + 3) * rhosolid
			S(3, j) = rb(j) * Pburn_loc(j + 3) * rhosolid * hreaction
		end do

		if (t_local < 0.35_dp) then
					  S(1, 1) = S(1, 1) + (4.0_dp/10.0_dp) * mig / dx_local
					  S(2, 1) = S(2, 1) + (4.0_dp/10.0_dp) * mig * vinj / dx_local
					  S(3, 1) = S(3, 1) + (4.0_dp/10.0_dp) * mig * hig / dx_local

					  S(1, 2) = S(1, 2) + (3.0_dp/10.0_dp) * mig / dx_local
					  S(2, 2) = S(2, 2) + (3.0_dp/10.0_dp) * mig * vinj / dx_local
					  S(3, 2) = S(3, 2) + (3.0_dp/10.0_dp) *  mig * hig / dx_local

					  S(1, 3) = S(1, 3) + (2.0_dp/10.0_dp) * mig / dx_local
					  S(2, 3) = S(2, 3) + (2.0_dp/10.0_dp) * mig * vinj / dx_local
					  S(3, 3) = S(3, 3) + (2.0_dp/10.0_dp) * mig * hig / dx_local

					  S(1, 4) = S(1, 4) + (1.0_dp/10.0_dp) * mig / dx_local
					  S(2, 4) = S(2, 4) + (1.0_dp/10.0_dp) * mig * vinj / dx_local
					  S(3, 4) = S(3, 4) + (1.0_dp/10.0_dp) * mig * hig / dx_local
		end if
	end subroutine source_term

		real(dp) function find_dt(U, A_loc, Pburn_loc, Pwet_loc, web_loc, table, &
			dx_local, cfl_local, t_local, k, ignited, erosive_on) result(dt_out)
			real(dp), intent(in) :: U(nvar, nx), A_loc(nx), Pburn_loc(nx), Pwet_loc(nx)
			real(dp), intent(in) :: web_loc(nx - 7), dx_local, cfl_local, t_local, k(nx)
			type(grain_geometry_table_t), intent(in) :: table
			logical, intent(in) :: ignited(nx - 7), erosive_on
			real(dp) :: SL(nx - 5), SR(nx - 5), maxabs
			real(dp) :: Sdt(3, nx - 6), rbdt(nx - 7), Gdt(nx - 7)
			real(dp) :: dt_convective, dt_source, dt_geometry, web_spacing
			real(dp) :: rho_value, velocity_value, pressure_value, sound_speed_value
			integer :: j

			call assert_admissible_state(U, A_loc, k, 'time-step calculation')
			call max_wave_speed_toro(U, A_loc, k, SL, SR, maxabs)
			do j = 1, nx
				rho_value = U(1, j) / A_loc(j)
				velocity_value = U(2, j) / U(1, j)
				pressure_value = (k(j) - 1.0_dp) * &
					(U(3, j) - 0.5_dp * U(2, j) * velocity_value) / A_loc(j)
				sound_speed_value = sqrt(k(j) * pressure_value / rho_value)
				maxabs = max(maxabs, abs(velocity_value) + sound_speed_value)
			end do
			if (.not. ieee_is_finite(maxabs) .or. maxabs <= 0.0_dp) then
				error stop 'Invalid maximum wave speed in find_dt.'
			end if
			dt_convective = cfl_local * dx_local / maxabs

			call source_term(U, A_loc, Pburn_loc, Pwet_loc, ignited, dx_local, &
				t_local, k, erosive_on, Sdt, rbdt, Gdt)
			dt_source = huge(1.0_dp)
			do j = 1, nx - 6
				if (Sdt(1, j) > 0.0_dp) then
					dt_source = min(dt_source, source_cfl * U(1, j + 3) / Sdt(1, j))
				end if
				if (Sdt(3, j) > 0.0_dp) then
					dt_source = min(dt_source, source_cfl * U(3, j + 3) / Sdt(3, j))
				end if
			end do

			dt_geometry = huge(1.0_dp)
			do j = 1, nx - 7
				if (rbdt(j) > 0.0_dp) then
					web_spacing = local_web_spacing(table, web_loc(j))
					dt_geometry = min(dt_geometry, geometry_cfl * web_spacing / rbdt(j))
				end if
			end do

			dt_out = min(dt_convective, dt_source, dt_geometry)
			if (.not. ieee_is_finite(dt_out) .or. dt_out <= 0.0_dp) then
				error stop 'Invalid time step after convective/source/geometry limits.'
			end if
		end function find_dt

	subroutine bound_cumulative_web(web_candidate, web_previous, table, web_bounded)
		real(dp), intent(in) :: web_candidate(nx - 7), web_previous(nx - 7)
		type(grain_geometry_table_t), intent(in) :: table
		real(dp), intent(out) :: web_bounded(nx - 7)
		real(dp) :: regression_tol

		if (any(.not. ieee_is_finite(web_candidate))) then
			error stop 'Nonfinite cumulative web in SSPRK geometry stage.'
		end if
		regression_tol = geometry_table_tol * max(1.0_dp, table%web_max)
		if (any(web_candidate < web_previous - regression_tol)) then
			error stop 'Cumulative web decreased in SSPRK geometry stage.'
		end if
		web_bounded = min(table%web_max, max(web_previous, web_candidate))
	end subroutine bound_cumulative_web

	subroutine geometry_from_web(web_inner, table, A_template, Pwet_template, &
		A_out, Pburn_out, Pwet_out)
		real(dp), intent(in) :: web_inner(nx - 7), A_template(nx), Pwet_template(nx)
		type(grain_geometry_table_t), intent(in) :: table
		real(dp), intent(out) :: A_out(nx), Pburn_out(nx), Pwet_out(nx)
		real(dp) :: area_inner(nx - 7), p_burn_inner(nx - 7), p_wet_inner(nx - 7)

		if (table%n_x /= nx - 7) then
			error stop 'Grain table size changed after validation.'
		end if
		call interpolate_grain_geometry(table, web_inner, area_inner, &
			p_burn_inner, p_wet_inner)

		A_out = A_template
		Pburn_out = Pwet_template
		Pwet_out = Pwet_template
		A_out(4:nx - 4) = area_inner
		Pburn_out(4:nx - 4) = p_burn_inner
		Pwet_out(4:nx - 4) = p_wet_inner
		call fill_geometry_ghosts(A_out, Pwet_out)
		call fill_perimeter_ghosts(Pburn_out)
	end subroutine geometry_from_web

	subroutine ssprk45(U, A_loc, Pburn_loc, Pwet_loc, web_loc, table, state, &
		dt_local, dx_local, t_local, k, erosive_on, Unp1, Anew, Pburnnew, &
		Pwetnew, webnew, rb_cycle, G_cycle)
		real(dp), intent(in) :: U(nvar, nx), A_loc(nx), Pburn_loc(nx), Pwet_loc(nx)
		real(dp), intent(in) :: web_loc(nx - 7), dt_local, dx_local, t_local, k(nx)
		type(grain_geometry_table_t), intent(in) :: table
		type(thermal_state_t), intent(inout) :: state
		logical, intent(in) :: erosive_on
		real(dp), intent(out) :: Unp1(nvar, nx), Anew(nx), Pburnnew(nx), Pwetnew(nx)
		real(dp), intent(out) :: webnew(nx - 7), rb_cycle(nx - 7), G_cycle(nx - 7)

		real(dp) :: U0(nvar, nx), U1(nvar, nx), U2(nvar, nx), U3(nvar, nx), U4(nvar, nx), kwork(nx)
		real(dp) :: A0(nx), A1(nx), A2(nx), A3(nx), A4(nx)
		real(dp) :: Pburn0(nx), Pburn1(nx), Pburn2(nx), Pburn3(nx), Pburn4(nx)
		real(dp) :: Pwet0(nx), Pwet1(nx), Pwet2(nx), Pwet3(nx), Pwet4(nx)
		real(dp) :: fm(3, nx - 6), fp(3, nx - 6), S(3, nx - 6)
		real(dp) :: k1(3, nx - 6), k2(3, nx - 6), k3(3, nx - 6), k4(3, nx - 6), k5(3, nx - 6)
		real(dp) :: rb1(nx - 7), rb2(nx - 7), rb3(nx - 7), rb4(nx - 7), rb5(nx - 7)
		real(dp) :: G1(nx - 7), G2(nx - 7), G3(nx - 7), G4(nx - 7), G5(nx - 7)
		real(dp) :: web0(nx - 7), web1(nx - 7), web2(nx - 7), web3(nx - 7), web4(nx - 7)
		real(dp) :: web_candidate(nx - 7)
		real(dp), parameter :: c2 = 0.391752226571890_dp
		real(dp), parameter :: c3 = 0.586079688967798_dp
		real(dp), parameter :: c4 = 0.474542363026874_dp
		real(dp), parameter :: c5 = 0.935010631009241_dp
		integer :: j

		U0 = U
		U1 = U
		U2 = U
		U3 = U
		U4 = U
		Unp1 = U
		kwork = k
		A0 = A_loc
		Pburn0 = Pburn_loc
		Pwet0 = Pwet_loc
		web0 = web_loc

		if (t_local < 0.35_dp) then
			call update_ignition_thermal_state(U0, A0, Pwet0, state, dt_local, kwork)
		else
			call report_and_force_ignition(state, t_local)
		end if

		! --- Stage 1 ---
		call riemann_bc(U0, A0, kwork, 'wall-atmosphere')
		call assert_admissible_state(U0, A0, kwork, 'SSPRK stage 1 input')

		t0 = omp_get_wtime()
		call hllc_flux(U0, A0, kwork, fm, fp)
		prof_hllc = prof_hllc + (omp_get_wtime() - t0)
		prof_call_hllc = prof_call_hllc + 1

		t0 = omp_get_wtime()
		call source_term(U0, A0, Pburn0, Pwet0, state%ignited, dx_local, &
			t_local, kwork, erosive_on, S, rb1, G1)
		prof_source = prof_source + (omp_get_wtime() - t0)
		prof_call_source = prof_call_source + 1

		k1 = -(fp - fm) / dx_local + S
		do j = 4, nx - 3
			U1(:, j) = U0(:, j) + 0.391752226571890_dp * dt_local * k1(:, j - 3)
		end do
		web_candidate = web0 + 0.391752226571890_dp * dt_local * rb1
		call bound_cumulative_web(web_candidate, web0, table, web1)
		call geometry_from_web(web1, table, A0, Pwet0, A1, Pburn1, Pwet1)
		call assert_admissible_state(U1, A1, kwork, 'SSPRK stage 1 output')

		! --- Stage 2 ---
		call riemann_bc(U1, A1, kwork, 'wall-atmosphere')
		call assert_admissible_state(U1, A1, kwork, 'SSPRK stage 2 input')
		t0 = omp_get_wtime()
		call hllc_flux(U1, A1, kwork, fm, fp)
		prof_hllc = prof_hllc + (omp_get_wtime() - t0)
		prof_call_hllc = prof_call_hllc + 1
		t0 = omp_get_wtime()
		call source_term(U1, A1, Pburn1, Pwet1, state%ignited, dx_local, &
			t_local + c2 * dt_local, kwork, erosive_on, S, rb2, G2)
		prof_source = prof_source + (omp_get_wtime() - t0)
		prof_call_source = prof_call_source + 1
		k2 = -(fp - fm) / dx_local + S
		do j = 4, nx - 3
			U2(:, j) = 0.444370493651235_dp * U0(:, j) + &
						  0.555629506348765_dp * U1(:, j) + &
						  0.368410593050371_dp * dt_local * k2(:, j - 3)
		end do
		web_candidate = 0.444370493651235_dp * web0 + 0.555629506348765_dp * web1 + &
			0.368410593050371_dp * dt_local * rb2
		call bound_cumulative_web(web_candidate, web0, table, web2)
		call geometry_from_web(web2, table, A0, Pwet0, A2, Pburn2, Pwet2)
		call assert_admissible_state(U2, A2, kwork, 'SSPRK stage 2 output')

		! --- Stage 3 ---
		call riemann_bc(U2, A2, kwork, 'wall-atmosphere')
		call assert_admissible_state(U2, A2, kwork, 'SSPRK stage 3 input')
		t0 = omp_get_wtime()
		call hllc_flux(U2, A2, kwork, fm, fp)
		prof_hllc = prof_hllc + (omp_get_wtime() - t0)
		prof_call_hllc = prof_call_hllc + 1
		t0 = omp_get_wtime()
		call source_term(U2, A2, Pburn2, Pwet2, state%ignited, dx_local, &
			t_local + c3 * dt_local, kwork, erosive_on, S, rb3, G3)
		prof_source = prof_source + (omp_get_wtime() - t0)
		prof_call_source = prof_call_source + 1
		k3 = -(fp - fm) / dx_local + S
		do j = 4, nx - 3
			U3(:, j) = 0.620101851488403_dp * U0(:, j) + &
						  0.379898148511597_dp * U2(:, j) + &
						  0.251891774271694_dp * dt_local * k3(:, j - 3)
		end do
		web_candidate = 0.620101851488403_dp * web0 + 0.379898148511597_dp * web2 + &
			0.251891774271694_dp * dt_local * rb3
		call bound_cumulative_web(web_candidate, web0, table, web3)
		call geometry_from_web(web3, table, A0, Pwet0, A3, Pburn3, Pwet3)
		call assert_admissible_state(U3, A3, kwork, 'SSPRK stage 3 output')

		! --- Stage 4 ---
		call riemann_bc(U3, A3, kwork, 'wall-atmosphere')
		call assert_admissible_state(U3, A3, kwork, 'SSPRK stage 4 input')
		t0 = omp_get_wtime()
		call hllc_flux(U3, A3, kwork, fm, fp)
		prof_hllc = prof_hllc + (omp_get_wtime() - t0)
		prof_call_hllc = prof_call_hllc + 1
		t0 = omp_get_wtime()
		call source_term(U3, A3, Pburn3, Pwet3, state%ignited, dx_local, &
			t_local + c4 * dt_local, kwork, erosive_on, S, rb4, G4)
		prof_source = prof_source + (omp_get_wtime() - t0)
		prof_call_source = prof_call_source + 1
		k4 = -(fp - fm) / dx_local + S
		do j = 4, nx - 3
			U4(:, j) = 0.178079954393132_dp * U0(:, j) + &
						  0.821920045606868_dp * U3(:, j) + &
						  0.544974750228521_dp * dt_local * k4(:, j - 3)
		end do
		web_candidate = 0.178079954393132_dp * web0 + 0.821920045606868_dp * web3 + &
			0.544974750228521_dp * dt_local * rb4
		call bound_cumulative_web(web_candidate, web0, table, web4)
		call geometry_from_web(web4, table, A0, Pwet0, A4, Pburn4, Pwet4)
		call assert_admissible_state(U4, A4, kwork, 'SSPRK stage 4 output')

		! --- Stage 5 ---
		call riemann_bc(U4, A4, kwork, 'wall-atmosphere')
		call assert_admissible_state(U4, A4, kwork, 'SSPRK stage 5 input')
		t0 = omp_get_wtime()
		call hllc_flux(U4, A4, kwork, fm, fp)
		prof_hllc = prof_hllc + (omp_get_wtime() - t0)
		prof_call_hllc = prof_call_hllc + 1
		t0 = omp_get_wtime()
		call source_term(U4, A4, Pburn4, Pwet4, state%ignited, dx_local, &
			t_local + c5 * dt_local, kwork, erosive_on, S, rb5, G5)
		prof_source = prof_source + (omp_get_wtime() - t0)
		prof_call_source = prof_call_source + 1
		k5 = -(fp - fm) / dx_local + S

		do j = 4, nx - 3
			Unp1(:, j) = 0.517231671970585_dp * U2(:, j) + 0.096059710526147_dp * U3(:, j) + &
				0.063692468666290_dp * dt_local * k4(:, j - 3) + &
				0.386708617503268_dp * U4(:, j) + &
				0.226007483236906_dp * dt_local * k5(:, j - 3)
		end do

		rb_cycle = 0.1468118760847865_dp * rb1 + 0.24848290944497606_dp * rb2 + 0.10425883033198079_dp * rb3 + &
							 0.27443890090135015_dp * rb4 + 0.226007483236906_dp * rb5

		G_cycle = 0.1468118760847865_dp * G1 + 0.24848290944497606_dp * G2 + 0.10425883033198079_dp * G3 + &
							 0.27443890090135015_dp * G4 + 0.226007483236906_dp * G5
		web_candidate = web0 + dt_local * rb_cycle
		call bound_cumulative_web(web_candidate, web0, table, webnew)
		call geometry_from_web(webnew, table, A0, Pwet0, Anew, Pburnnew, Pwetnew)
		! Report the accepted cumulative-web rate (including saturation at web_max).
		rb_cycle = (webnew - web0) / dt_local
		call assert_admissible_state(Unp1, Anew, kwork, 'SSPRK final output')
	end subroutine ssprk45

end program rocket_tester
