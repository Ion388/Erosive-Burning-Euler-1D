import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection

# ----------------- Utilities -----------------
def make_star(R=0.56433, eps=0, n=0, Npts=800):
    theta = np.linspace(0, 2*np.pi, Npts, endpoint=False)
    r = R * (1 + eps * np.cos(n * theta))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pts = np.stack([x, y], axis=1)

    # visualize point order along the boundary
    x = pts[:, 0]
    y = pts[:, 1]
    n = len(pts)

    plt.figure(figsize=(6, 6))
    # connect points in order (closed)
    plt.plot(np.append(x, x[0]), np.append(y, y[0]), '-k', lw=0.8, alpha=0.6)

    # color points by index
    sc = plt.scatter(x, y, c=np.arange(n), cmap='viridis', s=18, zorder=3)
    plt.colorbar(sc, label='point index')

    # arrows showing direction from each point to the next
    dx = np.roll(x, -1) - x
    dy = np.roll(y, -1) - y
    plt.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=1,
               width=0.003, color='C1', alpha=0.8, zorder=2)

    # annotate a subsample of indices so plot stays readable
    step = max(1, n // 50)
    for i in range(0, n, step):
        plt.text(x[i], y[i], str(i), fontsize=8, ha='center', va='center', color='white',
                 path_effects=[])
    theta_circle = np.linspace(0, 2*np.pi, 400)
    plt.plot(1.45 * np.cos(theta_circle), 1.45 * np.sin(theta_circle), 'r--', lw=1.5, label='radius 1.45')
    plt.axis('equal')
    plt.title('Point order along boundary (indices and arrows show sequence)')
    plt.show()

    return pts

def make_double_anchor(xlim=0.166, ylim=1.4, theta_spread=0.2*np.pi, rlim=1.45, Npts=1000):
    thetalim1 = np.pi/2 - theta_spread/2
    thetalim2 = np.pi/2 + theta_spread/2
    # xlim = xlim/3.3215
    # ylim = ylim/3.3215
    # rlim = rlim/3.3215
    xlin = np.linspace(-xlim, xlim, Npts//4)
    ylin = np.linspace(0, ylim, Npts//4)
    left_vert = np.stack([-xlim * np.ones_like(ylin), ylin], axis=1)
    right_vert = np.stack([xlim * np.ones_like(ylin), np.flip(ylin)], axis=1)

    
    theta1 = np.linspace(np.arctan(-ylim/xlim)+np.pi, thetalim2, Npts//4, endpoint=True)
    r1 = np.sqrt(xlim**2 + ylim**2)
    x1 = r1 * np.cos(theta1)
    y1 = r1 * np.sin(theta1)
    pts1 = np.stack([x1, y1], axis=1)
    theta2 = np.linspace(thetalim1, np.arctan(ylim/xlim), Npts//4, endpoint=True)
    x2 = r1 * np.cos(theta2)
    y2 = r1 * np.sin(theta2)
    pts2 = np.stack([x2, y2], axis=1)
    
    theta_outer = np.linspace(thetalim2, thetalim1, Npts//2, endpoint=True)
    xouter = rlim * np.cos(theta_outer)
    youter = rlim * np.sin(theta_outer)
    pts_outer = np.stack([xouter, youter], axis=1)
    n_bridge = Npts // 50
    # create 50 points strictly between last point of pts1 and first of pts_outer
    bridge1 = np.linspace(pts1[-1], pts_outer[0], n_bridge + 2, endpoint=True)[1:-1]
    bridge2 = np.linspace(pts_outer[-1], pts2[0], n_bridge + 2, endpoint=True)[1:-1]
    top = np.flip(np.vstack([left_vert, pts1, bridge1, pts_outer, bridge2, pts2, right_vert]), axis=0)
    # mirror top around x-axis: reverse order so the boundary stays continuous, then flip y
    bottom = top[::-1].copy()
    # remove first and last entries
    bottom = bottom[1:-1]
    bottom[:, 1] *= -1.0
    pts = np.vstack([top, (bottom)])
    

    #### color plots
    # visualize point order along the boundary
    x = pts[:, 0]
    y = pts[:, 1]
    n = len(pts)

    plt.figure(figsize=(6, 6))
    # connect points in order (closed)
    plt.plot(np.append(x, x[0]), np.append(y, y[0]), '-k', lw=0.8, alpha=0.6)

    # color points by index
    sc = plt.scatter(x, y, c=np.arange(n), cmap='viridis', s=18, zorder=3)
    plt.colorbar(sc, label='point index')

    # arrows showing direction from each point to the next
    dx = np.roll(x, -1) - x
    dy = np.roll(y, -1) - y
    plt.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=1,
               width=0.003, color='C1', alpha=0.8, zorder=2)

    # annotate a subsample of indices so plot stays readable
    step = max(1, n // 50)
    for i in range(0, n, step):
        plt.text(x[i], y[i], str(i), fontsize=8, ha='center', va='center', color='white',
                 path_effects=[])
    theta_circle = np.linspace(0, 2*np.pi, 400)
    plt.plot(1.45 * np.cos(theta_circle), 1.45 * np.sin(theta_circle), 'r--', lw=1.5, label='radius 1.45')
    plt.axis('equal')
    plt.title('Point order along boundary (indices and arrows show sequence)')
    plt.show()

    return pts

# make_double_anchor()

def make_circle(R=1.0, Npts=400):

    theta_out = np.linspace(0, 2*np.pi, Npts, endpoint=False)
    x_out = R * np.cos(theta_out)
    y_out = R * np.sin(theta_out)
    pts = np.stack([x_out, y_out], axis=1)


    # # visualize point order along the boundary (indices and direction)
    # x = pts[:, 0]; y = pts[:, 1]; n = len(pts)
    # plt.figure(figsize=(6,6))
    # plt.plot(np.append(x, x[0]), np.append(y, y[0]), '-k', lw=0.8, alpha=0.6)
    # sc = plt.scatter(x, y, c=np.arange(n), cmap='viridis', s=18, zorder=3)
    # plt.colorbar(sc, label='point index')
    # dx = np.roll(x, -1) - x
    # dy = np.roll(y, -1) - y
    # plt.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=1,
    #            width=0.003, color='C1', alpha=0.8, zorder=2)
    # step = max(1, n // 50)
    # for i in range(0, n, step):
    #     plt.text(x[i], y[i], str(i), fontsize=8, ha='center', va='center', color='white')
    # theta_circle = np.linspace(0, 2*np.pi, 400)
    # plt.plot(1.45 * np.cos(theta_circle), 1.45 * np.sin(theta_circle), 'r--', lw=1.5, label='radius 1.45')
    # plt.axis('equal')
    # plt.title('Circle with circular cavity (indices and arrows show sequence)')
    # plt.show()

    return pts

make_circle()

def area_shoelace(pts):
    # polygon area (positive)
    x = pts[:,0]
    y = pts[:,1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def perimeter(pts, R_max=1.45):
    pts = np.asarray(pts, dtype=float)
    if pts.size == 0:
        return 0.0
    # closed loop
    nxt = np.roll(pts, -1, axis=0)
    d = nxt - pts
    seg_lengths = np.hypot(d[:,0], d[:,1])
    total = seg_lengths.sum()

    # identify segments whose BOTH endpoints lie on the circle (within tolerance)
    x = pts[:,0]; y = pts[:,1]
    r = np.hypot(x, y)
    tol = 1e-3 # 0.1% tolerance
    on_circle = np.isclose(r, R_max, rtol=tol)

    # compute arc lengths for segments on the circle (use angular difference for accuracy)
    theta = np.arctan2(y, x)
    theta_nxt = np.roll(theta, -1)
    mask = on_circle & np.roll(on_circle, -1)
    if np.any(mask):
        dtheta = theta_nxt[mask] - theta[mask]
        # wrap to [-pi, pi]
        dtheta = (dtheta + np.pi) % (2*np.pi) - np.pi
        arc_lengths = R_max * np.abs(dtheta)
        # subtract arc lengths (ignore perimeter that lies on the circle)
        total -= arc_lengths.sum()

    return total

def unit_tangents_and_outward_normals(pts):
    """
    Returns (t_unit, normals) where normals are guaranteed to point OUTWARD
    from the polygon (away from its centroid).
    pts: array (N,2) ordered counterclockwise OR clockwise (we fix normals).
    """
    # central difference approximation for tangent (periodic)
    prev = np.roll(pts, 1, axis=0)
    nxt  = np.roll(pts, -1, axis=0)
    tang = nxt - prev                     # approx 2*ds * tangent
    tang_norm = np.hypot(tang[:,0], tang[:,1])
    tang_norm[tang_norm == 0] = 1.0
    t_unit = (tang.T / tang_norm).T       # unit tangent (signed by point order)
    # print(max(tang_norm))

    # rotate tangent by +90 deg to get an "outward"-candidate normal:
    normals = np.stack([-t_unit[:,1], t_unit[:,0]], axis=1)
    # print(max(normals.flatten()))
    
    # ensure outward: compare normal direction with vector from centroid to point
    centroid = pts.mean(axis=0)
    vec_from_centroid = pts - centroid   # points from center -> boundary
    dots = (normals * vec_from_centroid).sum(axis=1)  # positive if normal points outward
    # if dot < 0 then normal points inward: flip it
    inward_mask = dots < 0
    if np.any(inward_mask):
        normals[inward_mask] *= -1.0

    return t_unit, normals

def resample_by_arclength(pts, Nnew):
    # compute cumulative arclength
    closed = np.vstack([pts, pts[0]])
    d = np.diff(closed, axis=0)
    seglen = np.hypot(d[:,0], d[:,1])
    s = np.concatenate(([0.0], np.cumsum(seglen)))  # length N+1
    L = s[-1]
    # make periodic by dropping final repeated point for interpolation
    s_nodes = s[:-1]
    x_nodes = closed[:-1,0]
    y_nodes = closed[:-1,1]
    s_uniform = np.linspace(0.0, L, Nnew, endpoint=False)

    # use periodic cubic interpolation: extend one extra period for smoothness
    # add the first node at s=L
    s_ext = np.concatenate([s_nodes, [L]])
    x_ext = np.concatenate([x_nodes, [x_nodes[0]]])
    y_ext = np.concatenate([y_nodes, [y_nodes[0]]])

    fx = interp1d(s_ext, x_ext, kind='cubic')
    fy = interp1d(s_ext, y_ext, kind='cubic')
    new_x = fx(s_uniform)
    new_y = fy(s_uniform)
    return np.stack([new_x, new_y], axis=1)

# ----------------- Evolution -----------------
def step_normal_move(pts, Vn, dt):
    """
    Move points along OUTWARD normal by distance Vn*dt.
    Vn: scalar or array(len=N). Positive Vn --> port GROWS (outward).
    """
    t_unit, normals = unit_tangents_and_outward_normals(pts)
    if np.isscalar(Vn):
        move = normals * (Vn * dt)
    else:
        move = normals * ((Vn * dt)[:, None])
    pts_new = pts + move
    return pts_new

# def offset_port(pts, dr):
#     """
#     Offset polygon by distance dr and return a single clean boundary:
#     - if buffer produces multiple pieces, keep the largest polygon by area
#     - remove any interior holes (keep only the exterior ring)
#     - if buffer is empty or fails, return original pts
#     """
#     poly = Polygon(pts)
#     expanded = poly.buffer(+dr, join_style=2)  # 2 = mitre; 1=round; 3=bevel

#     if expanded.is_empty:
#         return np.asarray(pts, dtype=float)

#     # pick the largest polygon if we got multiple pieces / a collection
#     if isinstance(expanded, Polygon):
#         main = expanded
#     elif isinstance(expanded, (MultiPolygon, GeometryCollection)):
#         polys = [g for g in expanded.geoms if isinstance(g, Polygon)]
#         if not polys:
#             return np.asarray(pts, dtype=float)
#         main = max(polys, key=lambda p: p.area)
#     else:
#         return np.asarray(pts, dtype=float)

#     # return only the exterior boundary (drop holes / inner islands)
#     xs, ys = main.exterior.coords.xy
#     return np.column_stack([np.array(xs, dtype=float), np.array(ys, dtype=float)])

def subtract_intersections(pts):
    """
    Take a polygon (Nx2 numpy array or a shapely Polygon) and return a cleaned
    exterior boundary (numpy array Mx2) with self-intersecting / overlapping
    pieces removed. If multiple pieces result, the largest-by-area exterior is returned.
    If cleaning fails, the original polygon coordinates are returned.
    """
    # accept either numpy array of points or a shapely Polygon
    if isinstance(pts, Polygon):
        poly = pts
    else:
        pts = np.asarray(pts, dtype=float)
        if pts.size == 0:
            return pts
        poly = Polygon(pts)

    if poly.is_empty:
        return np.asarray(pts, dtype=float)

    # buffer(0) often fixes self-intersections and returns a valid geometry
    cleaned = poly.buffer(0)

    if cleaned.is_empty:
        return np.asarray(pts, dtype=float)

    # choose largest polygon if we got multiple pieces
    if isinstance(cleaned, Polygon):
        main = cleaned
    elif isinstance(cleaned, (MultiPolygon, GeometryCollection)):
        polys = [g for g in cleaned.geoms if isinstance(g, Polygon)]
        if not polys:
            return np.asarray(pts, dtype=float)
        main = max(polys, key=lambda p: p.area)
    else:
        return np.asarray(pts, dtype=float)

    xs, ys = main.exterior.coords.xy
    result = np.column_stack([np.array(xs, dtype=float), np.array(ys, dtype=float)])

    # drop duplicated closing point for consistency with other routines
    if result.shape[0] > 1 and np.allclose(result[0], result[-1]):
        result = result[:-1]

    return result

# ----------------- Demo / Animation -----------------
def simulate_front_tracking(R=1.0, eps=0.9, n=6, Npts=400,
                            V0=0.02, dt=1, steps=100, resample_N=400,
                            show_interval=30):
    """
    V0 : outward normal speed (units of length / time)
    dt : time step size (chosen to keep V0*dt small relative to segment length)
    steps : number of frames to animate (or iterations)
    resample_N : number of points to maintain by resampling
    show_interval : ms between frames for animation
    """
    # pts = make_star()
    # pts = make_double_anchor()
    pts1 = make_circle(R=0.83337, Npts=Npts)
    pts2 = make_circle(R=0.61663, Npts=Npts)


    fig, ax = plt.subplots(figsize=(6,6))
    # draw the initial polygon and the requested circle (radius 1.45 at origin)
    # ax.plot(pts[:, 0], pts[:, 1], '-k', lw=0.8, alpha=0.6)
    theta_circle = np.linspace(0, 2*np.pi, 400)
    ax.plot(1.45 * np.cos(theta_circle), 1.45 * np.sin(theta_circle), 'r--', lw=1.5, label='radius 1.45')
    line1, = ax.plot([], [], lw=1.2)
    line2, = ax.plot([], [], lw=1.2)
    # circ, = ax.plot([], [], lw=0.6, ls='--')  # showing best-fit circle optionally
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-1.5, 1.5*R)
    ax.set_ylim(-1.5*R, 1.5*R)
    ax.grid(True)

    txtA = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    txtP = ax.text(0.02, 0.90, '', transform=ax.transAxes)
    txtAP = ax.text(0.02, 0.85, '', transform=ax.transAxes)
    txtt = ax.text(0.02, 0.80, '', transform=ax.transAxes)

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        # circ.set_data([], [])
        txtA.set_text('')
        txtP.set_text('')
        txtAP.set_text('')
        txtt.set_text('')
        return line1, line2, txtA, txtP, txtAP, txtt

    def update(frame):
        nonlocal pts1, pts2
        # stability: take small internal steps
        n_sub = 200
        dt_sub = dt / n_sub
        R_max = 1.45  # max radius to allow outward movement
        for _ in range(n_sub):
            pts1 = step_normal_move(pts1, V0, dt_sub)
            pts2 = step_normal_move(pts2, -V0, dt_sub)
            # optionally stop outward movement if points go too far from center
            dists = np.hypot(pts1[:,0], pts1[:,1])
            # pts_max = pts.copy()
            pts1[dists > R_max, 0] = (R_max / dists[dists > R_max]) * pts1[dists > R_max, 0]
            pts1[dists > R_max, 1] = (R_max / dists[dists > R_max]) * pts1[dists > R_max, 1]
            # pts = np.minimum(pts, pts_max)

        # critical: re-sample to uniform arclength (this implements tangential motion)
        pts1 = np.asarray(pts1, dtype=float)
        pts2 = np.asarray(pts2, dtype=float)
        # remove duplicate closing point if present (first == last)
        if pts1.shape[0] > 1 and np.allclose(pts1[0], pts1[-1]):
            pts1 = pts1[:-1]
        pts1 = resample_by_arclength(pts1, Npts)
        if pts2.shape[0] > 1 and np.allclose(pts2[0], pts2[-1]):
            pts2 = pts2[:-1]
        pts2 = resample_by_arclength(pts2, Npts)

        x1 = np.append(pts1[:,0], pts1[0,0])
        y1 = np.append(pts1[:,1], pts1[0,1])
        line1.set_data(x1, y1)

        x2 = np.append(pts2[:,0], pts2[0,0])
        y2 = np.append(pts2[:,1], pts2[0,1])
        line2.set_data(x2, y2)

        A = area_shoelace(pts1) - area_shoelace(pts2)
        P = perimeter(pts1) + perimeter(pts2)
        txtA.set_text(f"A = {A:.6f}")
        txtP.set_text(f"P = {P:.6f}")
        txtAP.set_text(f"A/P = {A/P:.6f}")
        txtt.set_text(f"t = {frame * dt:.4f}")


        return line1, line2, txtA, txtP, txtAP, txtt
    anim = FuncAnimation(fig, update, frames=range(steps+1), init_func=init,
                         blit=True, interval=show_interval, repeat=False)
    plt.show()
    return anim

if __name__ == "__main__":
    # Choose dt so that V0*dt is << average segment length (for stability)
    # Rough heuristic: avg segment ~ perimeter/Npts, so dt < 0.2*(perim/Npts)/V0
    simulate_front_tracking(R=1.0, eps=0.95, n=6, Npts=2000,
                            V0=0.02/3, dt=0.1, steps=1100, resample_N=2000,
                            show_interval=2)
    
def area_perimeter_evolution(r=0.02/3*9/11, dt=0.1, steps=1100, resample_N=3000, n_sub=200):
    """
    Run a headless front-tracking loop (no animation) and plot area vs time.
    Parameters mirror simulate_front_tracking for easy comparison.
    """
    # pts = make_double_anchor()  # use same initializer as the animation demo
    pts1 = make_circle(R=0.83476, Npts=400)
    pts2 = make_circle(R=0.61524, Npts=400)

    areas = []
    perimeters = []
    hyd_diameters = []
    times = []
    

    for frame in range(steps + 1):
        # take small internal sub-steps (same stability strategy as animation)
        dt_sub = dt / float(n_sub)
        R_max = 1.45
        for _ in range(n_sub):
            pts1 = step_normal_move(pts1, r, dt_sub)
            pts2 = step_normal_move(pts2, -r, dt_sub)
            # optionally stop outward movement if points go too far from center
            dists = np.hypot(pts1[:,0], pts1[:,1])
            # pts_max = pts.copy()
            pts1[dists > R_max, 0] = (R_max / dists[dists > R_max]) * pts1[dists > R_max, 0]
            pts1[dists > R_max, 1] = (R_max / dists[dists > R_max]) * pts1[dists > R_max, 1]
            # pts = np.minimum(pts, pts_max)

        # critical: re-sample to uniform arclength (this implements tangential motion)
        pts1 = np.asarray(pts1, dtype=float)
        pts2 = np.asarray(pts2, dtype=float)
        # remove duplicate closing point if present (first == last)
        if pts1.shape[0] > 1 and np.allclose(pts1[0], pts1[-1]):
            pts1 = pts1[:-1]
        pts1 = resample_by_arclength(pts1, 400)
        if pts2.shape[0] > 1 and np.allclose(pts2[0], pts2[-1]):
            pts2 = pts2[:-1]
        pts2 = resample_by_arclength(pts2, 400)

        # compute area and record
        A = area_shoelace(pts1) - area_shoelace(pts2)
        P = perimeter(pts1) + perimeter(pts2)
        areas.append(A)
        perimeters.append(P)
        hyd_diameters.append(4 * A / P)
        times.append(frame * dt)
        if frame == 0:
            print(f"Initial area: {A:.6f}, perimeter: {P:.6f}, hydraulic diameter: {4*A/P:.6f}")

    t = np.array(times)
    A = np.array(areas)
    P = np.array(perimeters)
    D_h = np.array(hyd_diameters)

    print(f"Final area over initial area: {A[-1]/A[0]:.6f}")
    
    
    
    # split into two halves at the cutoff (keep original t and A intact)
    # t_cutoff = 26.0
    # mask1 = t <= t_cutoff
    # mask2 = t > t_cutoff
    # t1, A1, P1 = t[mask1], A[mask1], P[mask1]
    # t2, A2, P2 = t[mask2], A[mask2], P[mask2]

    # fit quadratic A(t) = a t^2 + b t + c
    # coeffs1 = np.polyfit(t1, A1, 1)  # [a, b, c]
    # a1, b1 = coeffs1
    # coeffs11 = np.polyfit(t1, P1, 1)  # [a, b, c]
    # a11, b11 = coeffs11

    # coeffs2 = np.polyfit(t2, A2, 2)  # [a, b, c]
    # a2, b2, c2 = coeffs2
    # coeffs22 = np.polyfit(t2, P2, 6)  # [a, b, c]
    # # a22, b22, c22 = coeffs22

    # # evaluate fitted polynomial on a fine grid (useful for plotting)
    # t_fit1 = np.linspace(t1.min(), t1.max(), 2000)
    # A_fit1 = np.polyval(coeffs1, t_fit1)
    # P_fit1 = np.polyval(coeffs11, t_fit1)

    # t_fit2 = np.linspace(t2.min(), t2.max(), 2000)
    # A_fit2 = np.polyval(coeffs2, t_fit2)
    # P_fit2 = np.polyval(coeffs22, t_fit2)

    # print(f"Linear fit: A(t) = {a1:.6e} t + {b1:.6e}")
    # print(f"Linear fit: P(t) = {a11:.6e} t + {b11:.6e}")
    # print(f"Quadratic fit: A(t) = {a2:.6e} t^2 + {b2:.6e} t + {c2:.6e}")
    # print(f"Quadratic fit: P(t) = {a22:.6e} t^2 + {b22:.6e} t + {c22:.6e}")

    # coeffs = np.polyfit(t, A, 4)  # [a, b, c, d, e]
    # t_fit = np.linspace(t.min(), t.max(), 2000)
    # A_fit = np.polyval(coeffs, t_fit)

    # # compute R^2 to assess goodness of fit
    # res = A - np.polyval(coeffs, t)
    # ss_res = np.sum(res**2)
    # ss_tot = np.sum((A - A.mean())**2)
    # r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan
    # print(f"R^2 of 4th-degree polynomial fit: {r2:.6f}")

    # plot results
    plt.figure(figsize=(6,4))
    plt.plot(times, areas, '-o', markersize=4)
    # plt.plot(t_fit1, A_fit1, '--r', label='Linear fit Area (first half)')
    # plt.plot(t_fit2, A_fit2, '--r', label='Quadratic fit Area(second half)')
    # plt.plot(t_fit, A_fit, '--r', label='4th-degree polynomial fit')
    plt.xlabel('time')
    plt.ylabel('area')
    plt.title('Evolution of surface area over time')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6,4))
    plt.plot(times, perimeters, '-o', markersize=4)
    # plt.plot(t_fit1, P_fit1, '--r', label='Linear fit Perimeter (first half)')
    # plt.plot(t_fit2, P_fit2, '--r', label='Quadratic fit Perimeter (second half)')
    plt.xlabel('time')
    plt.ylabel('perimeter')
    plt.title('Evolution of surface perimeter over time')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6,4))
    plt.plot(times, D_h, '-o', markersize=4)
    plt.xlabel('time')
    plt.ylabel('hydraulic diameter')
    plt.title('Evolution of hydraulic diameter over time')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return np.array(times), np.array(areas), np.array(perimeters), np.array(hyd_diameters)

def area_perimeter_helper(pts1, pts2, r=0.02/3, dt=0.1, resample_N=3000, n_sub=200, R_max=1.45):

    dt_sub = dt / float(n_sub)
    for _ in range(n_sub):
        pts1 = step_normal_move(pts1, r, dt_sub)
        dists1 = np.hypot(pts1[:,0], pts1[:,1])
        pts2 = step_normal_move(pts2,-r, dt_sub)
        pts1[dists1 > R_max, 0] = (R_max / dists1[dists1 > R_max]) * pts1[dists1 > R_max, 0]
        pts1[dists1 > R_max, 1] = (R_max / dists1[dists1 > R_max]) * pts1[dists1 > R_max, 1]

    # resample and clean intersections (same post-step processing as animation)
    pts1 = np.asarray(pts1, dtype=float)
    if pts1.shape[0] > 1 and np.allclose(pts1[0], pts1[-1]):
        pts1 = pts1[:-1]
    pts1 = resample_by_arclength(pts1, resample_N)

    pts2 = np.asarray(pts2, dtype=float)
    if pts2.shape[0] > 1 and np.allclose(pts2[0], pts2[-1]):
        pts2 = pts2[:-1]
    pts2 = resample_by_arclength(pts2, resample_N)

    # compute area and record
    A = area_shoelace(pts1) - area_shoelace(pts2)
    P = perimeter(pts1) + perimeter(pts2)
    D_h = 4 * A / P

    return pts1, pts2, A, P, D_h


# run the area plot after the animation (or on its own)
if __name__ == "__main__":
    area_perimeter_evolution()