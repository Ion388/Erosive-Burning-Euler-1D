import numpy as np
import os
import matplotlib.pyplot as plt

class LevelSet2D:
  def __init__(self):
    # Parameters
    self.nx, self.ny = 600, 600
    self.ng = 1
    self.xmin, self.xmax = -2.5, 2.5
    self.ymin, self.ymax = -2.5, 2.5
    self.rad_grain = 1.45
    self.rb = 0.01 # 10 mm/s
    
    self.t = 0.0
    self.t_end = 200.0
    self.dt = -1.0
    self.cfl = 0.5
    self.out_every = 1
    self.reinit_every = 20
    self.reinit_steps = 10
    self.reinit_cfl = 0.5
    self.n = 0
    
    self.dx = (self.xmax - self.xmin) / (self.nx - 1)
    self.dy = (self.ymax - self.ymin) / (self.ny - 1)
    self.eps0 = 1e-10 * max(1.0, min(self.dx, self.dy))
    
    self.x = np.linspace(self.xmin, self.xmax, self.nx)
    self.y = np.linspace(self.ymin, self.ymax, self.ny)
    self.xx, self.yy = np.meshgrid(self.x, self.y, indexing='ij')
    
    self.phi = np.zeros((self.nx + 2*self.ng, self.ny + 2*self.ng))
    self.F = np.zeros_like(self.phi)
    
    self.lfv = np.zeros(20001)
    self.lfa = np.zeros(20001)
    self.time = np.zeros(20001)
  
  def init_phi_flower(self):
    rg = 1.45
    r = np.sqrt(self.xx**2 + self.yy**2)
    teta = np.arctan2(self.yy, self.xx)
    rp0 = 0.46 * (1 + 0*np.cos(5.0*teta))
    self.phi[self.ng:self.nx+self.ng, self.ng:self.ny+self.ng] = np.where(rp0 > rg, rg - rp0, r - rp0)
    self.apply_neumann_bc()
  
  def fill_speed(self, t, mask):
    self.F[self.ng:self.nx+self.ng, self.ng:self.ny+self.ng] = np.where(mask, self.rb, 0.0)
    self.apply_neumann_bc_array(self.F)
  
  def apply_neumann_bc(self):
    self._apply_neumann_bc_impl(self.phi)
  
  def apply_neumann_bc_array(self, a):
    self._apply_neumann_bc_impl(a)
  
  def _apply_neumann_bc_impl(self, a):
    for g in range(1, self.ng+1):
      a[0:self.nx, self.ng-g] = a[0:self.nx, self.ng]
      a[0:self.nx, self.ny+self.ng+g-1] = a[0:self.nx, self.ny+self.ng-1]
      a[self.ng-g, :] = a[self.ng, :]
      a[self.nx+self.ng+g-1, :] = a[self.nx+self.ng-1, :]
  
  def advect_levelset_godunov(self):
    phi = self.phi
    F = self.F
    dx, dy = self.dx, self.dy
    ng, nx, ny = self.ng, self.nx, self.ny
    
    dpx = (phi[ng+1:nx+ng+1, ng:ny+ng] - phi[ng:nx+ng, ng:ny+ng]) / dx
    dmx = (phi[ng:nx+ng, ng:ny+ng] - phi[ng-1:nx+ng-1, ng:ny+ng]) / dx
    dpy = (phi[ng:nx+ng, ng+1:ny+ng+1] - phi[ng:nx+ng, ng:ny+ng]) / dy
    dmy = (phi[ng:nx+ng, ng:ny+ng] - phi[ng:nx+ng, ng-1:ny+ng-1]) / dy
    
    s = F[ng:nx+ng, ng:ny+ng]
    gradp2 = np.maximum(dmx, 0)**2 + np.minimum(dpx, 0)**2 + np.maximum(dmy, 0)**2 + np.minimum(dpy, 0)**2
    gradm2 = np.maximum(dpx, 0)**2 + np.minimum(dmx, 0)**2 + np.maximum(dpy, 0)**2 + np.minimum(dmy, 0)**2
    
    grad = np.where(s >= 0.0, np.sqrt(gradp2), np.sqrt(gradm2))
    self.phi[ng:nx+ng, ng:ny+ng] -= self.dt * s * grad
    self.apply_neumann_bc()

  def reinitialize_levelset(self, steps=None):
    if steps is None:
      steps = self.reinit_steps
    if steps <= 0:
      return

    phi = self.phi
    ng, nx, ny = self.ng, self.nx, self.ny
    dx, dy = self.dx, self.dy
    dtau = self.reinit_cfl * min(dx, dy)

    phi0 = phi[ng:nx+ng, ng:ny+ng].copy()
    sign_phi0 = phi0 / np.sqrt(phi0**2 + min(dx, dy)**2)

    for _ in range(steps):
      dpx = (phi[ng+1:nx+ng+1, ng:ny+ng] - phi[ng:nx+ng, ng:ny+ng]) / dx
      dmx = (phi[ng:nx+ng, ng:ny+ng] - phi[ng-1:nx+ng-1, ng:ny+ng]) / dx
      dpy = (phi[ng:nx+ng, ng+1:ny+ng+1] - phi[ng:nx+ng, ng:ny+ng]) / dy
      dmy = (phi[ng:nx+ng, ng:ny+ng] - phi[ng:nx+ng, ng-1:ny+ng-1]) / dy

      grad_plus = np.sqrt(
        np.maximum(dmx, 0.0)**2 + np.minimum(dpx, 0.0)**2 +
        np.maximum(dmy, 0.0)**2 + np.minimum(dpy, 0.0)**2
      )
      grad_minus = np.sqrt(
        np.minimum(dmx, 0.0)**2 + np.maximum(dpx, 0.0)**2 +
        np.minimum(dmy, 0.0)**2 + np.maximum(dpy, 0.0)**2
      )

      update = np.maximum(sign_phi0, 0.0) * (grad_plus - 1.0) + np.minimum(sign_phi0, 0.0) * (grad_minus - 1.0)
      phi[ng:nx+ng, ng:ny+ng] -= dtau * update
      self.apply_neumann_bc()
  
  def run(self):
    os.makedirs('vtk', exist_ok=True)
    self.init_phi_flower()
    
    mask = np.sqrt(self.xx**2 + self.yy**2) <= self.rad_grain
    
    delta_front = 1000.0
    
    with open('res.txt', 'w') as f:
      while self.t < self.t_end and delta_front > 1e-4:
        self.n += 1
        
        self.fill_speed(self.t, mask)
        Fmax = np.max(np.abs(self.F[self.ng:self.nx+self.ng, self.ng:self.ny+self.ng]))
        self.dt = 1e-6 if Fmax <= 1e-14 else self.cfl * min(self.dx, self.dy) / Fmax
        # dt = 0.1
        self.dt = min(self.dt, self.t_end - self.t)
        
        self.advect_levelset_godunov()

        if self.reinit_every > 0 and self.n % self.reinit_every == 0:
          self.reinitialize_levelset()

        self.t += self.dt
        
        phi_interior = self.phi[self.ng:self.nx+self.ng, self.ng:self.ny+self.ng]
        Lfront = self.extract_front_length(phi_interior)
        Afront = self.extract_area_inside(phi_interior)
        self.lfv[self.n] = Lfront
        self.lfa[self.n] = Afront
        delta_front = abs(Lfront - self.lfv[self.n-1]) if self.n > 0 else 1000.0
        
        if self.n % self.out_every == 0:
          print(f"step={self.n:7d}  t={self.t:10.6f}  dt={self.dt:10.6f}  L={Lfront:.4e}  delta={delta_front:.4e}  A={Afront:.4e}")
  
    
    print(f"Done. Steps: {self.n}, Final time: {self.t}")
    self.plot_and_save()
  
  def extract_front_length(self, phi):
    total_len = 0.0
    for j in range(self.ny-1):
      for i in range(self.nx-1):
        for xtri, ytri, ptri in [
          ([self.x[i], self.x[i+1], self.x[i+1]], 
           [self.y[j], self.y[j], self.y[j+1]], 
           [phi[i,j], phi[i+1,j], phi[i+1,j+1]]),
          ([self.x[i], self.x[i+1], self.x[i]], 
           [self.y[j], self.y[j+1], self.y[j+1]], 
           [phi[i,j], phi[i+1,j+1], phi[i,j+1]])
        ]:
          hasseg, x1, y1, x2, y2 = self.triangle_zero_segment(xtri, ytri, ptri)
          if hasseg:
            r1 = np.sqrt(x1**2 + y1**2)
            r2 = np.sqrt(x2**2 + y2**2)
            if r1 <= self.rad_grain and r2 <= self.rad_grain:
              total_len += np.sqrt((x2-x1)**2 + (y2-y1)**2)
    return total_len

  def extract_area_inside(self, phi):
    total_area = 0.0
    for j in range(self.ny-1):
      for i in range(self.nx-1):
        for xtri, ytri, ptri in [
          ([self.x[i], self.x[i+1], self.x[i+1]], 
          [self.y[j], self.y[j], self.y[j+1]], 
          [phi[i,j], phi[i+1,j], phi[i+1,j+1]]),
          ([self.x[i], self.x[i+1], self.x[i]], 
          [self.y[j], self.y[j+1], self.y[j+1]], 
          [phi[i,j], phi[i+1,j+1], phi[i,j+1]])
        ]:
          if np.all(np.array(ptri) < 0.0):
            tri_area = 0.5 * abs((xtri[1]-xtri[0])*(ytri[2]-ytri[0]) - (xtri[2]-xtri[0])*(ytri[1]-ytri[0]))
            total_area += tri_area
    return total_area
  
  def triangle_zero_segment(self, xv, yv, pv):
    xi, yi = [], []
    for a, b in [(0,1), (1,2), (2,0)]:
      pa, pb = float(pv[a]), float(pv[b])
      if abs(pa) < self.eps0: pa = 0.0
      if abs(pb) < self.eps0: pb = 0.0
      
      if pa == 0.0 and pb == 0.0: continue
      if pa == 0.0:
        self.add_unique_point(xv[a], yv[a], xi, yi, self.eps0)
      elif pb == 0.0:
        self.add_unique_point(xv[b], yv[b], xi, yi, self.eps0)
      elif pa*pb < 0.0:
        t = pa / (pa - pb)
        x = xv[a] + t*(xv[b] - xv[a])
        y = yv[a] + t*(yv[b] - yv[a])
        self.add_unique_point(x, y, xi, yi, self.eps0)
    
    return (True, xi[0], yi[0], xi[1], yi[1]) if len(xi) == 2 else (False, 0, 0, 0, 0)
  
  def add_unique_point(self, xp, yp, xi, yi, eps0):
    if not any((xp - x)**2 + (yp - y)**2 <= eps0**2 for x, y in zip(xi, yi)):
      xi.append(xp)
      yi.append(yp)

  def plot_and_save(self):
    # Extract valid data (ignore zero padding)
    valid_time = np.arange(self.n)*self.dt

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot front length
    ax1.plot(valid_time, self.lfv[:self.n], 'r-', linewidth=2)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Front Length')
    ax1.set_title('Front Length vs Time')
    ax1.grid(True, alpha=0.3)

    # Plot area
    ax2.plot(valid_time, self.lfa[:self.n], 'b-', linewidth=2)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Area Inside')
    ax2.set_title('Area vs Time')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('length_area_vs_time.png', dpi=400)
    plt.show()

    # save data to text file
    # first column is time*speed => distance covered by the front, second column is front length, third column is area inside
    with open('length_area_data.txt', 'w') as f:
      f.write('Distance\tFront Length\tArea Inside\n')
      for i in range(self.n):
        f.write(f"{valid_time[i]*self.rb:.6f}\t{self.lfv[i]:.6e}\t{self.lfa[i]:.6e}\n")

if __name__ == "__main__":
  sim = LevelSet2D()
  sim.run()



  

  