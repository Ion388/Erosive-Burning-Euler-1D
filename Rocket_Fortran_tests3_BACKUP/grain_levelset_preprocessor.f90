program grain_levelset_preprocessor
	! Example: gfortran -O3 -fopenmp input_vegae.f90 &
	!          grain_levelset_preprocessor.f90 -o grain_levelset_preprocessor
	use input_vegae_mod, only: dp, Lcase, Rcase
	use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
	implicit none

	integer, parameter :: nx_solver = 200
	integer, parameter :: nstations = nx_solver - 7
	integer, parameter :: ngrid = 321
	integer, parameter :: nweb = 501
	integer, parameter :: ntheta_metric = 8192
	integer, parameter :: ntheta_casing = 4096
	integer, parameter :: max_sweep_cycles = 32
	integer, parameter :: star_lobes = 5

	real(dp), parameter :: motor_length = Lcase
	real(dp), parameter :: casing_radius = Rcase
	real(dp), parameter :: circle_radius = 0.60_dp
	real(dp), parameter :: star_amplitude = 0.30_dp
	real(dp), parameter :: web_max = casing_radius
	real(dp), parameter :: grid_padding = 0.10_dp
	real(dp), parameter :: sweep_tolerance = 1.0e-10_dp
	character(len=*), parameter :: output_filename = 'grain_geometry_table.dat'
	character(len=*), parameter :: fields_output_filename = 'grain_levelset_fields.dat'

	real(dp) :: x_station(nstations), blend_station(nstations)
	real(dp) :: blend_shape(nstations)
	real(dp), allocatable :: area_shape(:, :), pburn_shape(:, :), pwet_shape(:, :)
	real(dp), allocatable :: levelset_fields(:, :, :)
	logical, allocatable :: burned_shape(:, :)
	integer :: shape_index(nstations), nshape
	integer :: field_station(3), field_shape(3), i, is, iw, output_unit, fields_unit, ios
	real(dp) :: dx_solver, dw

	if (circle_radius + star_amplitude >= casing_radius) then
		error stop 'The initial star touches or exceeds the casing radius.'
	end if
	if (ngrid < 101 .or. mod(ngrid, 2) == 0) then
		error stop 'ngrid must be an odd integer of at least 101.'
	end if
	if (nweb < 3 .or. web_max <= 0.0_dp) error stop 'Invalid web table resolution.'

	dx_solver = motor_length / real(nx_solver - 6, dp)
	dw = web_max / real(nweb - 1, dp)
	do i = 1, nstations
		x_station(i) = (real(i - 1, dp) + 0.5_dp) * dx_solver
		blend_station(i) = axial_shape_blend(x_station(i))
	end do

	call collect_unique_shapes(blend_station, blend_shape, shape_index, nshape)
	allocate(area_shape(nweb, nshape), pburn_shape(nweb, nshape), &
		pwet_shape(nweb, nshape), burned_shape(nweb, nshape))
	allocate(levelset_fields(ngrid, ngrid, 3))
	field_station = [1, (nstations + 1) / 2, nstations]
	field_shape = shape_index(field_station)

	! Each axial station is a two-dimensional level-set problem. Repeated circle
	! and star stations share one calculation; only transition stations are unique.
	!$omp parallel do default(shared) private(is) schedule(dynamic)
	do is = 1, nshape
		if (is == field_shape(1)) then
			call evolve_shape(blend_shape(is), dw, area_shape(:, is), &
				pburn_shape(:, is), pwet_shape(:, is), burned_shape(:, is), levelset_fields(:, :, 1))
		else if (is == field_shape(2)) then
			call evolve_shape(blend_shape(is), dw, area_shape(:, is), &
				pburn_shape(:, is), pwet_shape(:, is), burned_shape(:, is), levelset_fields(:, :, 2))
		else if (is == field_shape(3)) then
			call evolve_shape(blend_shape(is), dw, area_shape(:, is), &
				pburn_shape(:, is), pwet_shape(:, is), burned_shape(:, is), levelset_fields(:, :, 3))
		else
			call evolve_shape(blend_shape(is), dw, area_shape(:, is), &
				pburn_shape(:, is), pwet_shape(:, is), burned_shape(:, is))
		end if
	end do
	!$omp end parallel do

	open(newunit=output_unit, file=output_filename, status='replace', action='write', iostat=ios)
	if (ios /= 0) error stop 'Could not open the grain geometry table for writing.'
	write(output_unit, '(A)') '# GRAIN_LEVELSET_TABLE_V1'
	write(output_unit, '(2(I0,1X),ES24.14E3)') nstations, nweb, casing_radius
	write(output_unit, '(A)') '# x w A_port P_burn P_wet burned_out'
	do i = 1, nstations
		is = shape_index(i)
		do iw = 1, nweb
			write(output_unit, '(5(ES24.14E3,1X),I1)') x_station(i), &
				real(iw - 1, dp) * dw, area_shape(iw, is), &
				pburn_shape(iw, is), pwet_shape(iw, is), &
				merge(1, 0, burned_shape(iw, is))
		end do
	end do
	close(output_unit)

	open(newunit=fields_unit, file=fields_output_filename, status='replace', action='write', iostat=ios)
	if (ios /= 0) error stop 'Could not open the level-set field file for writing.'
	write(fields_unit, '(A)') '# GRAIN_LEVELSET_FIELDS_V1'
	write(fields_unit, '(2(I0,1X),2(ES24.14E3,1X))') 3, ngrid, &
		-(casing_radius + grid_padding), 2.0_dp * (casing_radius + grid_padding) / real(ngrid - 1, dp)
	write(fields_unit, '(A)') '# station_x followed by ngrid rows of phi values'
	dw_solver: do i = 1, 3
		is = field_station(i)
		write(fields_unit, '(ES24.14E3)') x_station(is)
		do iw = 1, ngrid
			write(fields_unit, '(*(ES24.14E3,1X))') levelset_fields(:, iw, i)
		end do
	end do dw_solver
	close(fields_unit)

	write(*, '(A,I0)') 'Unique cross-sectional level-set evolutions: ', nshape
	write(*, '(A,ES14.6)') 'Web spacing: ', dw
	write(*, '(A,A)') 'Wrote ', output_filename
	write(*, '(A,A)') 'Wrote ', fields_output_filename

contains

	pure real(dp) function axial_shape_blend(x) result(value)
		real(dp), intent(in) :: x
		real(dp) :: xi, x1, x2

		x1 = motor_length / 3.0_dp
		x2 = 2.0_dp * motor_length / 3.0_dp
		if (x <= x1) then
			value = 0.0_dp
		else if (x >= x2) then
			value = 1.0_dp
		else
			xi = (x - x1) / (x2 - x1)
			value = xi * xi * xi * (10.0_dp + xi * (-15.0_dp + 6.0_dp * xi))
		end if
	end function axial_shape_blend

	pure real(dp) function initial_radius(blend, theta) result(radius)
		real(dp), intent(in) :: blend, theta

		! Edit this function to introduce a different analytical, star-shaped port.
		radius = circle_radius + blend * star_amplitude * &
			cos(real(star_lobes, dp) * theta)
	end function initial_radius

	subroutine collect_unique_shapes(blend_values, unique_values, indices, n_unique)
		real(dp), intent(in) :: blend_values(:)
		real(dp), intent(out) :: unique_values(size(blend_values))
		integer, intent(out) :: indices(size(blend_values)), n_unique
		integer :: j, k
		logical :: found

		n_unique = 0
		unique_values = 0.0_dp
		do j = 1, size(blend_values)
			found = .false.
			do k = 1, n_unique
				if (abs(blend_values(j) - unique_values(k)) <= 1.0e-14_dp) then
					indices(j) = k
					found = .true.
					exit
				end if
			end do
			if (.not. found) then
				n_unique = n_unique + 1
				unique_values(n_unique) = blend_values(j)
				indices(j) = n_unique
			end if
		end do
	end subroutine collect_unique_shapes

	subroutine evolve_shape(blend, dw_local, area, p_burn, p_wet, burned, levelset_field)
		real(dp), intent(in) :: blend, dw_local
		real(dp), intent(out) :: area(nweb), p_burn(nweb), p_wet(nweb)
		logical, intent(out) :: burned(nweb)
		real(dp), intent(out), optional :: levelset_field(:, :)
		real(dp), allocatable :: raw(:, :), distance(:, :), phi(:, :), p_raw(:)
		real(dp) :: half_width, h, xmin, xpos, ypos, radius2, eps_delta
		real(dp) :: theta, dtheta, phi_value, z, delta_value, web_value
		real(dp) :: initial_area, initial_perimeter, casing_area, w_full
		real(dp) :: active_integral, perimeter_scale, casing_contact
		integer :: ix, iy, k, m, mlo, mhi, burn_index

		half_width = casing_radius + grid_padding
		h = 2.0_dp * half_width / real(ngrid - 1, dp)
		xmin = -half_width
		eps_delta = 2.0_dp * h
		casing_area = acos(-1.0_dp) * casing_radius * casing_radius
		allocate(raw(ngrid, ngrid), distance(ngrid, ngrid), &
			phi(ngrid, ngrid), p_raw(nweb))

		do iy = 1, ngrid
			ypos = xmin + real(iy - 1, dp) * h
			do ix = 1, ngrid
				xpos = xmin + real(ix - 1, dp) * h
				theta = atan2(ypos, xpos)
				raw(ix, iy) = sqrt(xpos * xpos + ypos * ypos) - &
					initial_radius(blend, theta)
			end do
		end do

		! The Eikonal fast sweep is used only to initialize a true signed-distance
		! level set. For unit normal speed, phi(x,y,w)=phi(x,y,0)-w, so every
		! requested web contour is obtained without a second evolution method.
		call signed_distance_fast_sweep(raw, h, distance)
		where (raw <= 0.0_dp)
			phi = -distance
		elsewhere
			phi = distance
		end where
		if (present(levelset_field)) levelset_field = phi

		w_full = 0.0_dp
		do iy = 1, ngrid
			ypos = xmin + real(iy - 1, dp) * h
			do ix = 1, ngrid
				xpos = xmin + real(ix - 1, dp) * h
				if (xpos * xpos + ypos * ypos <= casing_radius * casing_radius) then
					w_full = max(w_full, phi(ix, iy))
				end if
			end do
		end do
		dtheta = 2.0_dp * acos(-1.0_dp) / real(ntheta_casing, dp)
		do m = 1, ntheta_casing
			theta = (real(m, dp) - 0.5_dp) * dtheta
			xpos = casing_radius * cos(theta)
			ypos = casing_radius * sin(theta)
			w_full = max(w_full, bilinear_value(phi, xmin, h, xpos, ypos))
		end do
		if (w_full >= web_max) error stop 'web_max does not reach complete burnout.'
		burn_index = min(nweb, max(2, ceiling(w_full / dw_local) + 1))

		call polar_shape_metrics(blend, initial_area, initial_perimeter)
		p_raw = 0.0_dp
		do iy = 1, ngrid
			ypos = xmin + real(iy - 1, dp) * h
			do ix = 1, ngrid
				xpos = xmin + real(ix - 1, dp) * h
				radius2 = xpos * xpos + ypos * ypos
				if (radius2 > casing_radius * casing_radius) cycle
				phi_value = phi(ix, iy)
				mlo = max(0, ceiling((phi_value - eps_delta) / dw_local))
				mhi = min(nweb - 1, floor((phi_value + eps_delta) / dw_local))
				do m = mlo, mhi
					z = phi_value - real(m, dp) * dw_local
					if (abs(z) <= eps_delta) then
						delta_value = 0.5_dp * (1.0_dp + &
							cos(acos(-1.0_dp) * z / eps_delta)) / eps_delta
						p_raw(m + 1) = p_raw(m + 1) + delta_value * h * h
					end if
				end do
			end do
		end do
		p_raw(1) = initial_perimeter
		p_raw(burn_index:nweb) = 0.0_dp

		active_integral = integrate_to_web(p_raw, dw_local, &
			real(burn_index - 1, dp) * dw_local)
		if (active_integral <= 0.0_dp) error stop 'Invalid level-set perimeter integral.'
		perimeter_scale = (casing_area - initial_area) / active_integral
		if (.not. ieee_is_finite(perimeter_scale) .or. &
			perimeter_scale < 0.80_dp .or. perimeter_scale > 1.20_dp) then
			error stop 'Level-set grid is too coarse: perimeter normalization is excessive.'
		end if

		do k = 1, nweb
			if (k < burn_index) then
				p_burn(k) = max(perimeter_scale * p_raw(k), 0.0_dp)
				burned(k) = .false.
			else
				p_burn(k) = 0.0_dp
				burned(k) = .true.
			end if
		end do

		area(1) = initial_area
		do k = 2, nweb
			if (k >= burn_index) then
				area(k) = casing_area
			else
				area(k) = min(casing_area, area(k - 1) + &
					0.5_dp * dw_local * (p_burn(k - 1) + p_burn(k)))
			end if
		end do

		p_wet = p_burn
		do k = 1, nweb
			if (burned(k)) then
				p_wet(k) = 2.0_dp * acos(-1.0_dp) * casing_radius
				cycle
			end if
			web_value = real(k - 1, dp) * dw_local
			casing_contact = 0.0_dp
			do m = 1, ntheta_casing
				theta = (real(m, dp) - 0.5_dp) * dtheta
				xpos = casing_radius * cos(theta)
				ypos = casing_radius * sin(theta)
				phi_value = bilinear_value(phi, xmin, h, xpos, ypos)
				casing_contact = casing_contact + &
					regularized_heaviside(web_value - phi_value, eps_delta)
			end do
			p_wet(k) = p_burn(k) + casing_radius * dtheta * casing_contact
		end do

		if (any(.not. ieee_is_finite(area)) .or. any(.not. ieee_is_finite(p_burn)) .or. &
			any(.not. ieee_is_finite(p_wet))) then
			error stop 'Nonfinite entry in generated geometry table.'
		end if
		deallocate(raw, distance, phi, p_raw)
	end subroutine evolve_shape

	subroutine polar_shape_metrics(blend, area0, perimeter0)
		real(dp), intent(in) :: blend
		real(dp), intent(out) :: area0, perimeter0
		real(dp) :: theta, dtheta, radius, radius_prev, radius_next, drdtheta
		integer :: m

		area0 = 0.0_dp
		perimeter0 = 0.0_dp
		dtheta = 2.0_dp * acos(-1.0_dp) / real(ntheta_metric, dp)
		do m = 1, ntheta_metric
			theta = (real(m, dp) - 0.5_dp) * dtheta
			radius = initial_radius(blend, theta)
			radius_prev = initial_radius(blend, theta - dtheta)
			radius_next = initial_radius(blend, theta + dtheta)
			drdtheta = (radius_next - radius_prev) / (2.0_dp * dtheta)
			area0 = area0 + 0.5_dp * radius * radius * dtheta
			perimeter0 = perimeter0 + sqrt(radius * radius + drdtheta * drdtheta) * dtheta
		end do
	end subroutine polar_shape_metrics

	subroutine signed_distance_fast_sweep(raw, h, distance)
		real(dp), intent(in) :: raw(:, :), h
		real(dp), intent(out) :: distance(size(raw, 1), size(raw, 2))
		real(dp) :: value1, value2, denom, d1, max_change
		integer :: ix, iy, cycle

		distance = huge(1.0_dp) / 16.0_dp
		do iy = 1, size(raw, 2)
			do ix = 1, size(raw, 1)
				if (raw(ix, iy) == 0.0_dp) distance(ix, iy) = 0.0_dp
			end do
			do ix = 1, size(raw, 1) - 1
				value1 = raw(ix, iy)
				value2 = raw(ix + 1, iy)
				if (value1 * value2 <= 0.0_dp) then
					denom = abs(value1) + abs(value2)
					if (denom > tiny(1.0_dp)) then
						d1 = h * abs(value1) / denom
						distance(ix, iy) = min(distance(ix, iy), d1)
						distance(ix + 1, iy) = min(distance(ix + 1, iy), h - d1)
					end if
				end if
			end do
		end do
		do ix = 1, size(raw, 1)
			do iy = 1, size(raw, 2) - 1
				value1 = raw(ix, iy)
				value2 = raw(ix, iy + 1)
				if (value1 * value2 <= 0.0_dp) then
					denom = abs(value1) + abs(value2)
					if (denom > tiny(1.0_dp)) then
						d1 = h * abs(value1) / denom
						distance(ix, iy) = min(distance(ix, iy), d1)
						distance(ix, iy + 1) = min(distance(ix, iy + 1), h - d1)
					end if
				end if
			end do
		end do

		do cycle = 1, max_sweep_cycles
			max_change = 0.0_dp
			call sweep_pass(distance, h, 1, ngrid, 1, 1, ngrid, 1, max_change)
			call sweep_pass(distance, h, ngrid, 1, -1, 1, ngrid, 1, max_change)
			call sweep_pass(distance, h, 1, ngrid, 1, ngrid, 1, -1, max_change)
			call sweep_pass(distance, h, ngrid, 1, -1, ngrid, 1, -1, max_change)
			if (max_change <= sweep_tolerance * h) exit
		end do
		if (any(distance >= huge(1.0_dp) / 32.0_dp)) then
			error stop 'Fast sweeping did not initialize the full distance field.'
		end if
	end subroutine signed_distance_fast_sweep

	subroutine sweep_pass(distance, h, ilo, ihi, istep, jlo, jhi, jstep, max_change)
		real(dp), intent(inout) :: distance(:, :), max_change
		real(dp), intent(in) :: h
		integer, intent(in) :: ilo, ihi, istep, jlo, jhi, jstep
		real(dp) :: old_value, new_value, a, b, discriminant, large_value
		integer :: ix, iy

		large_value = huge(1.0_dp) / 64.0_dp
		do iy = jlo, jhi, jstep
			do ix = ilo, ihi, istep
				a = large_value
				b = large_value
				if (ix > 1) a = min(a, distance(ix - 1, iy))
				if (ix < size(distance, 1)) a = min(a, distance(ix + 1, iy))
				if (iy > 1) b = min(b, distance(ix, iy - 1))
				if (iy < size(distance, 2)) b = min(b, distance(ix, iy + 1))
				if (a >= large_value .and. b >= large_value) cycle
				if (a >= large_value) then
					new_value = b + h
				else if (b >= large_value) then
					new_value = a + h
				else if (abs(a - b) >= h) then
					new_value = min(a, b) + h
				else
					discriminant = max(2.0_dp * h * h - (a - b) * (a - b), 0.0_dp)
					new_value = 0.5_dp * (a + b + sqrt(discriminant))
				end if
				old_value = distance(ix, iy)
				if (new_value < old_value) then
					distance(ix, iy) = new_value
					max_change = max(max_change, old_value - new_value)
				end if
			end do
		end do
	end subroutine sweep_pass

	pure real(dp) function integrate_to_web(values, spacing, limit) result(integral)
		real(dp), intent(in) :: values(:), spacing, limit
		real(dp) :: w_left, w_right, segment_right, fraction, value_right
		integer :: k

		integral = 0.0_dp
		do k = 1, size(values) - 1
			w_left = real(k - 1, dp) * spacing
			w_right = real(k, dp) * spacing
			if (w_left >= limit) exit
			segment_right = min(w_right, limit)
			fraction = (segment_right - w_left) / spacing
			value_right = values(k) + fraction * (values(k + 1) - values(k))
			integral = integral + 0.5_dp * (values(k) + value_right) * &
				(segment_right - w_left)
		end do
	end function integrate_to_web

	pure real(dp) function regularized_heaviside(argument, epsilon) result(value)
		real(dp), intent(in) :: argument, epsilon

		if (argument <= -epsilon) then
			value = 0.0_dp
		else if (argument >= epsilon) then
			value = 1.0_dp
		else
			value = 0.5_dp * (1.0_dp + argument / epsilon + &
				sin(acos(-1.0_dp) * argument / epsilon) / acos(-1.0_dp))
		end if
	end function regularized_heaviside

	pure real(dp) function bilinear_value(field, xmin, h, xpos, ypos) result(value)
		real(dp), intent(in) :: field(:, :), xmin, h, xpos, ypos
		real(dp) :: gx, gy, tx, ty
		integer :: ix, iy

		gx = (xpos - xmin) / h + 1.0_dp
		gy = (ypos - xmin) / h + 1.0_dp
		ix = max(1, min(size(field, 1) - 1, floor(gx)))
		iy = max(1, min(size(field, 2) - 1, floor(gy)))
		tx = max(0.0_dp, min(1.0_dp, gx - real(ix, dp)))
		ty = max(0.0_dp, min(1.0_dp, gy - real(iy, dp)))
		value = (1.0_dp - tx) * (1.0_dp - ty) * field(ix, iy) + &
			tx * (1.0_dp - ty) * field(ix + 1, iy) + &
			(1.0_dp - tx) * ty * field(ix, iy + 1) + &
			tx * ty * field(ix + 1, iy + 1)
	end function bilinear_value

end program grain_levelset_preprocessor
