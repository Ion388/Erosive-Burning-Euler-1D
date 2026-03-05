!=============================================================
! 2D Level-Set equation on a uniform structured grid (Fortran 90)
! phi_t + F(x,y,t) * |grad phi| = 0
! First-order Godunov upwind scheme, explicit time stepping
! VTK legacy ASCII output for ParaView (STRUCTURED_POINTS)
!
! Burn front: phi = 0 contour. Choose speed F as burn rate (m/s).
!=============================================================
program levelset2d_vtk
  implicit none
  integer, parameter :: rk = selected_real_kind(12, 200)
  real(rk), parameter :: pi=4.0_rk*datan(1.d0)
  integer :: nx, ny, ng
  real(rk) :: xmin, xmax, ymin, ymax, dx, dy, rad_grain, rr, rb
  real(rk) :: t, t_end, dt, cfl
  integer :: n, nsteps, out_every, reinit_every
  real(rk) :: Fmax

  real(rk), allocatable :: phi(:,:), phin(:,:), F(:,:)
  real(rk), allocatable :: segx(:), segy(:)
  real(rk) :: eps0, Lfront, Afront
  integer :: nseg
  logical, allocatable :: mask(:,:)
  real(rk), allocatable :: x(:), y(:)
  real(rk) :: lfv(0:20000), time(0:20000), lfa(0:20000)
  real(rk) :: delta_front, teta, dtet
  integer :: i,j,ip,ic
  integer, parameter :: ncircle=50

  !-------------------------
  ! User parameters
  !-------------------------
  nx   = 201
  ny   = 201
  ng   = 1                  ! ghost cells (1 is enough for 1st order)
  xmin = -2.5_rk; xmax = 2.5_rk   ! meters (example)
  ymin = -2.5_rk; ymax = 2.5_rk
  rad_grain = 1.45_rk
  rb = 0.02/3.0_rk             ! m/s (example burn rate)

  t      = 0.0_rk
  time(0)=t
  t_end  = 200.0_rk          ! seconds (example)
  cfl    = 0.5_rk
  out_every    = 5          ! output frequency in time steps
  reinit_every = 1000           ! set >0 to occasionally reinitialize (optional)

  !-------------------------
  ! Grid
  !-------------------------
  dx = (xmax - xmin) / real(nx-1, rk)
  dy = (ymax - ymin) / real(ny-1, rk)
  eps0 = 1.0e-10_rk * max(1.0_rk, min(dx,dy))

  allocate(x(nx), y(ny))
  call fill_coords(nx, xmin, dx, x)
  call fill_coords(ny, ymin, dy, y)

  ! Arrays include ghost layers: (1-ng : nx+ng)
  allocate(phi(1-ng:nx+ng, 1-ng:ny+ng))
  allocate(phin(1-ng:nx+ng,1-ng:ny+ng))
  allocate(F(1-ng:nx+ng,  1-ng:ny+ng))
  allocate(mask(1:nx, 1:ny))

  !-------------------------
  ! Initialize level set phi
  ! Example: circular port/burn front (phi=0 at r=R0)
  ! Sign convention: phi < 0 inside burned region, phi > 0 in solid
  !-------------------------
  call init_phi_flower(nx, ny, ng, x, y, phi, 1.45_rk)

  dtet = 2.d0*pi/ncircle

  open(5,file='res.txt')

  ! Initial VTK
  !call write_vtk_scalar("phi_00000.vtk", nx, ny, xmin, ymin, dx, dy, phi, "phi")
  call extract_front_segments_and_length(phi(1:nx,1:ny), rad_grain, x, y, nx, ny, eps0, segx, segy, nseg, Lfront, Afront)
  lfv(0)=Lfront
  lfa(0)=Afront
  if (nseg > 0) then
    call write_vtk_polydata_segments("front_00000.vtk", segx, segy, nseg)
    write(5,*)'# t=',t
    do ip = 1, nseg
      write(5,*)segx(2*ip-1),segy(2*ip-1)
      write(5,*)segx(2*ip),segy(2*ip)
      write(5,*)
    enddo
    do ic = 1, ncircle
      teta = (ic-1.0)*dtet
      write(5,*)rad_grain*cos(teta), rad_grain*sin(teta)
      teta = ic*dtet
      write(5,*)rad_grain*cos(teta), rad_grain*sin(teta)
      write(5,*)
    enddo
    write(5,*)
  endif
  if (allocated(segx)) deallocate(segx)
  if (allocated(segy)) deallocate(segy)
  write(*,'(A,ES12.4)') "Initial front length L = ", Lfront
  n = 0


  mask(:,:)=.false.
  do j = 1, ny
    do i = 1, nx
      rr=sqrt(x(i)**2+y(j)**2)
      if (rr<=rad_grain) mask(i,j)=.true.
    enddo
  enddo


  !-------------------------
  ! Time stepping
  !-------------------------
  nsteps = 0
  delta_front=1000.0_rk
  do while (t < t_end .and. delta_front>1.e-4_rk)
    nsteps = nsteps + 1

    ! Define front propagation speed F (m/s).
    ! For a "certain speed", use constant burn rate rb.
    call fill_speed(nx, ny, ng, x, y, t, F, mask, rb)  ! example: 4 mm/s

    Fmax = maxval(abs(F(1:nx,1:ny)))
    if (Fmax <= 1.0e-14_rk) then
      dt = 1.0e-6_rk
    else
      dt = cfl * min(dx, dy) / Fmax
    end if
    if (t + dt > t_end) dt = t_end - t

    call apply_neumann_bc(nx, ny, ng, phi)

    call advect_levelset_godunov(nx, ny, ng, dx, dy, dt, phi, F, phin)

    phi = phin
    t   = t + dt
    n   = n + 1
    time(n)=t

    ! Optional: reinitialize phi to signed distance occasionally
    if (reinit_every > 0) then
      if (mod(n, reinit_every) == 0) call reinitialize_sussman(nx, ny, ng, dx, dy, 30, 0.3_rk, phi)
    endif


    call extract_front_segments_and_length(phi(1:nx,1:ny), rad_grain, x, y, nx, ny, eps0, segx, segy, nseg, Lfront, Afront)
    lfv(n)=Lfront
    lfa(n)=Afront
    if (nseg > 0) then
      call write_vtk_polydata_segments(step_filename("front_", n, ".vtk"), segx, segy, nseg)
      write(5,*)'# t=',t
      do ip = 1, nseg
        write(5,*)segx(2*ip-1),segy(2*ip-1)
        write(5,*)segx(2*ip),segy(2*ip)
        write(5,*)
      enddo
      do ic = 1, ncircle
        teta = (ic-1.0)*dtet
        write(5,*)rad_grain*cos(teta), rad_grain*sin(teta)
        teta = ic*dtet
        write(5,*)rad_grain*cos(teta), rad_grain*sin(teta)
        write(5,*)
      enddo
      write(5,*) ! block separator: two empty lines

    endif
    if (allocated(segx)) deallocate(segx)
    if (allocated(segy)) deallocate(segy)


    if (mod(n, out_every) == 0) then
      !call write_vtk_scalar(step_filename("phi_", n, ".vtk"), nx, ny, xmin, ymin, dx, dy, phi, "phi")
      !call write_vtk_scalar(step_filename("F_", n, ".vtk"),   nx, ny, xmin, ymin, dx, dy, F,   "F")
      write(*,'(A,I7,A,F10.6,A,F10.6,A,2ES12.4)') "step=", n, "  t=", t, "  dt=", dt, "  L=", Lfront,delta_front
    endif

    delta_front=abs(lfv(n)-lfv(n-1))


  enddo

  close(5) ! gnuplot contour animation


  ! Final output
  !call write_vtk_scalar(step_filename("phi_", n, ".vtk"), nx, ny, xmin, ymin, dx, dy, phi, "phi")
  !call write_vtk_scalar(step_filename("F_",   n, ".vtk"), nx, ny, xmin, ymin, dx, dy, F, "F")
  call extract_front_segments_and_length(phi(1:nx,1:ny), rad_grain, x, y, nx, ny, eps0, segx, segy, nseg, Lfront, Afront)
  if (nseg > 0) call write_vtk_polydata_segments(step_filename("front_", n, ".vtk"), segx, segy, nseg)
  if (allocated(segx)) deallocate(segx)
  if (allocated(segy)) deallocate(segy)

  write(*,*) "Done. Steps:", n, " Final time:", t


  open(7,file='gnp-anim.txt')
  write(7,*)'set terminal gif animate delay  0.01'
  write(7,*)"set output 'contour.gif'"
  write(7,*)'set size ratio -1'
  write(7,*)'set xrange [-2.5 : 2.5]'
  write(7,*)'set yrange [-2.5 : 2.5 ]'
  write(7,*)'set grid'
  write(7,*)"set xlabel 'x [m]';"
  write(7,*)"set ylabel 'w [m]';"
  write(7,*)'N=',n
  write(7,*)'delay=',0.01
  write(7,*)'datafile="res.txt"'
  write(7,*)'do for [i=1:N] {'
  !write(7,*)"set title sprintf("frame %d", i)"
  write(7,*)'plot datafile index i using 1:2 with lines lw 2 notitle'
  write(7,*)'pause delay'
  write(7,*)'}'
  close(7)



  open(8,file='length.txt')
  do i=0,n
    write(8,*)time(i),lfv(i),lfa(i)
  enddo
  close(8)

  call system('gnuplot < gnp.txt')
  call system('gnuplot < gnp-anim.txt')

contains

  subroutine fill_coords(n, x0, dx, arr)
    integer, intent(in) :: n
    real(rk), intent(in) :: x0, dx
    real(rk), intent(out) :: arr(n)
    integer :: i
    do i = 1, n
      arr(i) = x0 + real(i-1, rk) * dx
    end do
  end subroutine fill_coords

  subroutine init_phi_flower(nx, ny, ng, x, y, phi, rg)
    integer, intent(in) :: nx, ny, ng
    real(rk), intent(in) :: x(nx), y(ny), rg
    real(rk), intent(out) :: phi(1-ng:nx+ng, 1-ng:ny+ng)
    integer :: i, j
    real(rk) :: r, rp0, teta
    do j = 1, ny
      do i = 1, nx
        r = sqrt(x(i)**2 + y(j)**2)
        !phi(i,j) = r - R0  ! phi=0 at initial front radius R0
        teta=atan2(y(j),x(i))
        rp0=0.46_rk*(1.0_rk+0.65_rk*cos(5.0_rk*teta))
        phi(i,j) = r - Rp0  ! phi=0 at initial front radius R0
        if (rp0>rg) phi(i,j)=rg-rp0
      enddo
    enddo
    call apply_neumann_bc(nx, ny, ng, phi)
  end subroutine init_phi_flower

  subroutine fill_speed(nx, ny, ng, x, y, t, F, mask, rb)
    integer, intent(in) :: nx, ny, ng
    real(rk), intent(in) :: x(nx), y(ny), t, rb
    real(rk), intent(out) :: F(1-ng:nx+ng, 1-ng:ny+ng)
    logical, intent(in) :: mask(nx,ny)
    integer :: i, j
    real(rk) :: rr
    ! Constant normal burn rate rb everywhere:
    do j = 1, ny
      do i = 1, nx
        rr=sqrt(x(i)**2+y(j)**2)
        if (mask(i,j)) then
          F(i,j) = rb
        else
          F(i,j) = 0.0_rk
        endif
      enddo
    enddo
    call apply_neumann_bc(nx, ny, ng, F)
  end subroutine fill_speed

  subroutine apply_neumann_bc(nx, ny, ng, a)
    integer, intent(in) :: nx, ny, ng
    real(rk), intent(inout) :: a(1-ng:nx+ng, 1-ng:ny+ng)
    integer :: g, i, j
    ! Left/right
    do j = 1, ny
      do g = 1, ng
        a(1-g, j)    = a(1, j)
        a(nx+g, j)   = a(nx, j)
      end do
    end do
    ! Bottom/top
    do i = 1, nx
      do g = 1, ng
        a(i, 1-g)    = a(i, 1)
        a(i, ny+g)   = a(i, ny)
      end do
    end do
    ! Corners (copy nearest interior)
    do g = 1, ng
      a(1-g, 1-g)       = a(1,1)
      a(1-g, ny+g)      = a(1,ny)
      a(nx+g, 1-g)      = a(nx,1)
      a(nx+g, ny+g)     = a(nx,ny)
    end do
  end subroutine apply_neumann_bc

  subroutine advect_levelset_godunov(nx, ny, ng, dx, dy, dt, phi, F, phin)
    integer, intent(in) :: nx, ny, ng
    real(rk), intent(in) :: dx, dy, dt
    real(rk), intent(in) :: phi(1-ng:nx+ng, 1-ng:ny+ng)
    real(rk), intent(in) :: F(1-ng:nx+ng,   1-ng:ny+ng)
    real(rk), intent(out):: phin(1-ng:nx+ng,1-ng:ny+ng)

    integer :: i, j
    real(rk) :: dpx, dmx, dpy, dmy
    real(rk) :: gradp2, gradm2, grad, s

    phin = phi

    do j = 1, ny
      do i = 1, nx
        dpx = (phi(i+1,j) - phi(i,j)) / dx
        dmx = (phi(i,j)   - phi(i-1,j)) / dx
        dpy = (phi(i,j+1) - phi(i,j)) / dy
        dmy = (phi(i,j)   - phi(i,j-1)) / dy

        s = F(i,j)

        if (s >= 0.0_rk) then
          ! Godunov (Osher-Sethian) for positive speed
          gradp2 = max(dmx, 0.0_rk)**2 + min(dpx, 0.0_rk)**2 &
                 + max(dmy, 0.0_rk)**2 + min(dpy, 0.0_rk)**2
          grad = sqrt(gradp2)
        else
          ! Godunov for negative speed
          gradm2 = max(dpx, 0.0_rk)**2 + min(dmx, 0.0_rk)**2 &
                 + max(dpy, 0.0_rk)**2 + min(dmy, 0.0_rk)**2
          grad = sqrt(gradm2)
        endif

        phin(i,j) = phi(i,j) - dt * s * grad
      enddo
    enddo

    call apply_neumann_bc(nx, ny, ng, phin)
  end subroutine advect_levelset_godunov

  !-----------------------------------------------------------
  ! Optional reinitialization (Sussman et al.) to keep |grad phi| ~ 1
  ! phi_tau + sign(phi0) (|grad phi| - 1) = 0
  !-----------------------------------------------------------
  subroutine reinitialize_sussman(nx, ny, ng, dx, dy, niter, dtau_fac, phi)
    integer, intent(in) :: nx, ny, ng, niter
    real(rk), intent(in) :: dx, dy, dtau_fac
    real(rk), intent(inout) :: phi(1-ng:nx+ng, 1-ng:ny+ng)

    integer :: k, i, j
    real(rk) :: dtau, eps, s0
    real(rk), allocatable :: phi0(:,:), tmp(:,:)
    real(rk) :: dpx, dmx, dpy, dmy, grad, g2

    allocate(phi0(1-ng:nx+ng,1-ng:ny+ng))
    allocate(tmp(1-ng:nx+ng, 1-ng:ny+ng))
    phi0 = phi

    dtau = dtau_fac * min(dx, dy)
    eps  = min(dx, dy)

    do k = 1, niter
      call apply_neumann_bc(nx, ny, ng, phi)

      do j = 1, ny
        do i = 1, nx
          s0 = phi0(i,j) / sqrt(phi0(i,j)**2 + eps**2)

          dpx = (phi(i+1,j) - phi(i,j)) / dx
          dmx = (phi(i,j)   - phi(i-1,j)) / dx
          dpy = (phi(i,j+1) - phi(i,j)) / dy
          dmy = (phi(i,j)   - phi(i,j-1)) / dy

          if (s0 >= 0.0_rk) then
            g2 = max(dmx,0.0_rk)**2 + min(dpx,0.0_rk)**2 + &
                 max(dmy,0.0_rk)**2 + min(dpy,0.0_rk)**2
          else
            g2 = max(dpx,0.0_rk)**2 + min(dmx,0.0_rk)**2 + &
                 max(dpy,0.0_rk)**2 + min(dmy,0.0_rk)**2
          end if
          grad = sqrt(g2)

          tmp(i,j) = phi(i,j) - dtau * s0 * (grad - 1.0_rk)
        end do
      end do

      phi(1:nx,1:ny) = tmp(1:nx,1:ny)
    end do

    deallocate(phi0, tmp)
  end subroutine reinitialize_sussman

  !-----------------------------------------------------------
  ! VTK legacy ASCII writer for uniform grid
  !-----------------------------------------------------------
  subroutine write_vtk_scalar(filename, nx, ny, x0, y0, dx, dy, a, scalname)
    character(len=*), intent(in) :: filename, scalname
    integer, intent(in) :: nx, ny
    real(rk), intent(in) :: x0, y0, dx, dy
    real(rk), intent(in) :: a(0:,0:)   ! we will pass a slice with 1:nx,1:ny
    integer :: i, j, npts, unit

    npts = nx * ny
    open(newunit=unit, file='vtk/'//filename, status="replace", action="write", form="formatted")

    write(unit,'(A)') "# vtk DataFile Version 3.0"
    write(unit,'(A)') "2D level set scalar"
    write(unit,'(A)') "ASCII"
    write(unit,'(A)') "DATASET STRUCTURED_POINTS"
    write(unit,'(A,3(I0,1X))') "DIMENSIONS ", nx, ny, 1
    write(unit,'(A,3(ES24.16,1X))') "ORIGIN ", x0, y0, 0.0_rk
    write(unit,'(A,3(ES24.16,1X))') "SPACING ", dx, dy, 1.0_rk
    write(unit,'(A,I0)') "POINT_DATA ", npts
    write(unit,'(A,A)') "SCALARS ", trim(scalname)//" float"
    write(unit,'(A)') "LOOKUP_TABLE default"

    ! VTK expects x fastest, then y. Write j outer, i inner.
    do j = 1, ny
      do i = 1, nx
        write(unit,'(ES24.16)') real(a(i,j), kind=rk)
      end do
    end do

    close(unit)
  end subroutine write_vtk_scalar


  !-----------------------------------------------------------
  ! Extract phi=0 contour as line segments by splitting each quad
  ! into two triangles, and write as VTK POLYDATA. Also returns
  ! total contour length (sum of segment lengths).
  !
  ! Assumptions:
  ! - phi is POINT data on a structured grid with coordinates x(i), y(j)
  ! - We process quads (i=1..nx-1, j=1..ny-1) using vertices:
  !     (i,j), (i+1,j), (i+1,j+1), (i,j+1)
  !   split into triangles: (00,10,11) and (00,11,01)
  !-----------------------------------------------------------
  subroutine extract_front_segments_and_length(phi, rad_grain, x, y, nx, ny, eps0, segx, segy, nseg, total_len, area)
    implicit none
    integer, intent(in) :: nx, ny
    real(rk), intent(in) :: phi(1:nx,1:ny), x(nx), y(ny), eps0, rad_grain
    real(rk), allocatable, intent(out) :: segx(:), segy(:)   ! packed endpoints length=2*nseg
    integer, intent(out) :: nseg
    real(rk), intent(out) :: total_len
    real(rk), intent(out) :: area

    integer :: i, j, maxseg, ip
    real(rk) :: xtri(3), ytri(3), ptri(3)
    real(rk) :: a1x, a1y, a2x, a2y
    logical :: hasseg, rad
    real(rk) :: len
    real(rk), allocatable :: wx(:), wy(:)   ! working arrays (stride 4 per segment)

    maxseg = 2 * (nx-1) * (ny-1)
    allocate(wx(4*maxseg))
    allocate(wy(4*maxseg))

    nseg = 0
    total_len = 0.0_rk
    area = 0.0_rk

    do j = 1, ny-1
      do i = 1, nx-1

        ! Triangle 1: (i,j) (i+1,j) (i+1,j+1)
        xtri = (/ x(i),     x(i+1),   x(i+1) /)
        ytri = (/ y(j),     y(j),     y(j+1) /)
        ptri = (/ phi(i,j), phi(i+1,j), phi(i+1,j+1) /)
        call triangle_zero_segment(xtri, ytri, ptri, eps0, hasseg, a1x, a1y, a2x, a2y)
        if (hasseg) then
          nseg = nseg + 1
          ip = 4*(nseg-1)
          wx(ip+1) = a1x; wy(ip+1) = a1y
          wx(ip+2) = a2x; wy(ip+2) = a2y
          wx(ip+3) = 0.0_rk; wy(ip+3) = 0.0_rk  ! unused
          wx(ip+4) = 0.0_rk; wy(ip+4) = 0.0_rk  ! unused
          rad=sqrt(a1x**2+a1y**2)<=rad_grain .and. sqrt(a2x**2+a2y**2)<=rad_grain
          if (rad) then
            len = sqrt((a2x-a1x)**2 + (a2y-a1y)**2)
            total_len = total_len + len
          endif
        endif

        ! Triangle 2: (i,j) (i+1,j+1) (i,j+1)
        xtri = (/ x(i),     x(i+1),     x(i) /)
        ytri = (/ y(j),     y(j+1),     y(j+1) /)
        ptri = (/ phi(i,j), phi(i+1,j+1), phi(i,j+1) /)
        call triangle_zero_segment(xtri, ytri, ptri, eps0, hasseg, a1x, a1y, a2x, a2y)
        if (hasseg) then
          nseg = nseg + 1
          ip = 4*(nseg-1)
          wx(ip+1) = a1x; wy(ip+1) = a1y
          wx(ip+2) = a2x; wy(ip+2) = a2y
          wx(ip+3) = 0.0_rk; wy(ip+3) = 0.0_rk
          wx(ip+4) = 0.0_rk; wy(ip+4) = 0.0_rk
          rad=sqrt(a1x**2+a1y**2)<=rad_grain .and. sqrt(a2x**2+a2y**2)<=rad_grain
          if (rad) then
            len = sqrt((a2x-a1x)**2 + (a2y-a1y)**2)
            total_len = total_len + len
          endif
        endif

        ! Compute area where phi < 0 (burned region)
        if (phi(i,j) < 0.0_rk .or. phi(i+1,j) < 0.0_rk .or. &
            phi(i+1,j+1) < 0.0_rk .or. phi(i,j+1) < 0.0_rk) then
          area = area + compute_burned_quad_area(x(i), y(j), x(i+1), y(j+1), &
                                                              phi(i,j), phi(i+1,j), &
                                                              phi(i+1,j+1), phi(i,j+1))
        endif

      enddo
    enddo

    ! Pack to endpoints-only arrays (length = 2*nseg)
    if (nseg > 0) then
      allocate(segx(2*nseg))
      allocate(segy(2*nseg))
      do ip = 1, nseg
        segx(2*ip-1) = wx(4*(ip-1)+1)
        segy(2*ip-1) = wy(4*(ip-1)+1)
        segx(2*ip  ) = wx(4*(ip-1)+2)
        segy(2*ip  ) = wy(4*(ip-1)+2)
      end do
    else
      allocate(segx(0))
      allocate(segy(0))
    end if

    deallocate(wx, wy)
    
    write(*,'(A,ES12.4)') "Inner area = ", area
  end subroutine extract_front_segments_and_length

  function compute_burned_quad_area(x0, y0, x1, y1, p00, p10, p11, p01) result(area)
    implicit none
    real(rk), intent(in) :: x0, y0, x1, y1, p00, p10, p11, p01
    real(rk) :: area
    real(rk) :: dx, dy, quad_area
    
    dx = x1 - x0
    dy = y1 - y0
    quad_area = dx * dy
    
    ! If all corners are burned (phi < 0), entire quad contributes
    if (p00 < 0.0_rk .and. p10 < 0.0_rk .and. p11 < 0.0_rk .and. p01 < 0.0_rk) then
      area = quad_area
    else
      ! Approximate: fraction based on average phi value
      ! Smaller (more negative) phi means more burned
      area = quad_area * max(0.0_rk, 1.0_rk + (p00 + p10 + p11 + p01) / (4.0_rk * max(abs(p00), abs(p10), abs(p11), abs(p01))))
      area = max(0.0_rk, area)
    endif
  end function compute_burned_quad_area


  subroutine triangle_zero_segment(xv, yv, pv, eps0, hasseg, x1, y1, x2, y2)
    implicit none
    ! Returns a single segment for the zero level set within one triangle
    ! using linear interpolation on edges. If the zero set does not cut the
    ! triangle (or is degenerate), hasseg=.false.
    real(rk), intent(in) :: xv(3), yv(3), pv(3), eps0
    logical, intent(out) :: hasseg
    real(rk), intent(out) :: x1, y1, x2, y2

    integer :: e
    integer, parameter :: ea(3) = (/1,2,3/)
    integer, parameter :: eb(3) = (/2,3,1/)
    real(rk) :: pa, pb, t, xi(3), yi(3)
    integer :: npt

    npt = 0
    xi = 0.0_rk; yi = 0.0_rk

    do e = 1, 3
      pa = pv(ea(e))
      pb = pv(eb(e))

      ! Snap near-zero to zero to reduce jitter
      if (abs(pa) < eps0) pa = 0.0_rk
      if (abs(pb) < eps0) pb = 0.0_rk

      if (pa == 0.0_rk .and. pb == 0.0_rk) cycle  ! entire edge on interface (degenerate)

      if (pa == 0.0_rk) then
        call add_unique_point(xv(ea(e)), yv(ea(e)), xi, yi, npt, eps0)
      else if (pb == 0.0_rk) then
        call add_unique_point(xv(eb(e)), yv(eb(e)), xi, yi, npt, eps0)
      else if (pa*pb < 0.0_rk) then
        t = pa / (pa - pb)   ! in (0,1)
        call add_unique_point(xv(ea(e)) + t*(xv(eb(e)) - xv(ea(e))), &
                              yv(ea(e)) + t*(yv(eb(e)) - yv(ea(e))), xi, yi, npt, eps0)
      end if
    end do

    if (npt == 2) then
      hasseg = .true.
      x1 = xi(1); y1 = yi(1)
      x2 = xi(2); y2 = yi(2)
    else
      hasseg = .false.
      x1 = 0.0_rk; y1 = 0.0_rk
      x2 = 0.0_rk; y2 = 0.0_rk
    end if
  end subroutine triangle_zero_segment


  subroutine add_unique_point(xp, yp, xi, yi, npt, eps0)
    implicit none
    real(rk), intent(in) :: xp, yp, eps0
    real(rk), intent(inout) :: xi(3), yi(3)
    integer, intent(inout) :: npt
    integer :: k
    real(rk) :: d2

    do k = 1, npt
      d2 = (xp - xi(k))**2 + (yp - yi(k))**2
      if (d2 <= (eps0*eps0)) return
    end do

    if (npt < 3) then
      npt = npt + 1
      xi(npt) = xp
      yi(npt) = yp
    end if
  end subroutine add_unique_point


  subroutine write_vtk_polydata_segments(filename, segx, segy, nseg)
    implicit none
    character(len=*), intent(in) :: filename
    real(rk), intent(in) :: segx(:), segy(:)   ! length = 2*nseg
    integer, intent(in) :: nseg
    integer :: unit, i, npts, k

    npts = 2*nseg
    open(newunit=unit, file='vtk/'//filename, status="replace", action="write", form="formatted")

    write(unit,'(A)') "# vtk DataFile Version 3.0"
    write(unit,'(A)') "phi=0 front segments"
    write(unit,'(A)') "ASCII"
    write(unit,'(A)') "DATASET POLYDATA"
    write(unit,'(A,1X,I0,1X,A)') "POINTS", npts, "double"

    do i = 1, npts
      write(unit,'(3(ES24.16,1X))') segx(i), segy(i), 0.0_rk
    end do

    write(unit,'(A,1X,I0,1X,I0)') "LINES", nseg, 3*nseg
    do k = 0, nseg-1
      write(unit,'(I0,1X,I0,1X,I0)') 2, 2*k, 2*k+1
    end do

    close(unit)
  end subroutine write_vtk_polydata_segments


  function step_filename(prefix, step, suffix) result(name)
    character(len=*), intent(in) :: prefix, suffix
    integer, intent(in) :: step
    character(len=256) :: name
    character(len=16) :: s
    write(s,'(I0)') step
    name = trim(prefix)//trim(adjustl(zpad(step,5)))//trim(suffix)
  end function step_filename

  function zpad(i, w) result(str)
    integer, intent(in) :: i, w
    character(len=32) :: str
    character(len=32) :: tmp
    integer :: nsp
    write(tmp,'(I0)') i
    nsp = w - len_trim(tmp)
    if (nsp < 0) nsp = 0
    str = repeat("0", nsp)//trim(tmp)
  end function zpad

end program levelset2d_vtk