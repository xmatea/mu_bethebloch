import numpy as np
from matplotlib import pyplot as plt

# universal constants (should do this in nat units really!)
e_0 = 8.85418 * 1e-12
m_e = 9.10938 * 1e-31
c = 2.99892 * 1e8
N_A = 6.02214 * 1e23
M_u = 1
e = 1.60218 * 1e-19
pi = np.pi

m_mu  = 1.8835 * 1e-28

def electron_number_density(Z, rho, A):
  return ((N_A * Z * rho) / (A * M_u)) * 1e6

def p_to_E(p, m):
  return np.sqrt((p*c)**2 + (m*c**2)**2) - m*c*c

def beta2_to_E(beta2, m):
  return m*c*c*np.sqrt(1+beta2)

def E_to_beta2(E, m):
  return (E*E)/(m*c*c)**2 - 1

def bethe_bloch(n, I, z, beta2):

  A = 4*pi /(m_e*c**2)
  B = (e**2/(4*pi*e_0))**2
  C = 2 * m_e * c**2
  D = n*z*z

  return - (A*B*D / beta2) * np.log((C*beta2)/(I*(1 - beta2))) + A*B*D


## 1D simulations
class Layer1D:
  def __init__(self, name: str, thickness: float, density: float):
    """
    Creates a new layer.

    Parameters
    ----------
    name: str - Name of the layer
    thickness: float - Layer thickness in mm
    density: float - Density of the layer in g/cm^3
    """

    self.density = density
    self.thickness = thickness/1e3
    self.name = name

    self.n = None
    self.I = None

  def material_from_mass_ratios(self, Zs: list[float], As: list[float], mass_ratios: list[float]):
    mass_ratios = np.array(mass_ratios)/np.sum(mass_ratios) # renormalise

    # calculate weighted average Z and electron density (very rough estimate!)
    n = 0
    weighted_Z = 0

    for i, Z in enumerate(Zs):
      n += electron_number_density(Z, self.density, As[i]) * mass_ratios[i]
      weighted_Z += Z * mass_ratios[i]

    self.n = n
    self.I = 10 * e * weighted_Z

  def material_from_Z_A(self, Z: int, A: float):
    self.n = electron_number_density(Z, self.density, A)
    self.I = 10 * e * Z

class Sample1D:
  def __init__(self):
    self.layers = []
    self.total_thickness = 0

  def add_layer(self, layer):
    self.layers.append(layer)
    self.total_thickness += layer.thickness


def run_1D_sim(sample, mom, dx):
  mom_SI = mom*1e6*e/c
  dx = dx/1e3

  E_0 = m_mu*c*c
  E = np.sqrt((mom_SI*c)**2 + (m_mu*c*c)**2)

  x = np.arange(0, sample.total_thickness, dx)

  E_s = np.zeros_like(x)

  i = 0
  for l, layer in enumerate(sample.layers):
    n = layer.n
    I = layer.I
    z = 1

    for _ in range(int(layer.thickness/dx)):
      beta_v = E_to_beta2(E, m_mu)

      E += bethe_bloch(n, I, z, beta_v) * dx
      E_s[i] = E - E_0

      if abs(E-E_0) <= 1e-14:
        depth = x[E_s==np.min(E_s)][0]
        print(f"Muon stopped at {depth*1e3:.4f} mm.")
        return E_s, x, depth, sample

      i += 1

  depth = 0
  print(f"Muon did not stop in sample.")
  print(f"last E={E}, E-E0={E-E_0}, E0={E_0}")
  return E_s, x, depth, sample


def plot_1D_sim(E_s, x, sample, xmin=None, xmax=None, shift=0):
  fig, ax = plt.subplots()

  ax.plot(x*1e3-shift, E_s/(e*1e6))

  if xmin is not None:
    ax.set_xlim(xmin=xmin)
  else:
    xmin = ax.get_xlim()[0]

  if xmax is not None:
    ax.set_xlim(xmax=xmax)
  else:
    xmax = ax.get_xlim()[1]

  thickness = 0
  for i, layer in enumerate(sample.layers):
    thickness += layer.thickness

    if xmax >= thickness*1e3-shift >= xmin:
      ax.axvline(x=thickness*1e3-shift, color='k', linestyle='--')
      ax.text(thickness*1e3-shift, np.max(E_s) * 0.75, f"{layer.name} end", horizontalalignment='left', rotation='vertical')

  ax.set_xlabel("Depth (mm)")
  ax.set_ylabel("Muon energy (MeV)")
  plt.suptitle("Bethe-Bloch simulation")

  plt.show()

class Beam2D:
  def __init__(self):
    pass

class Sample2D:
  def __init__(self, mask):
    pass

class Material:
  def __init__(self, name: str, density: float):
      self.n = None
      self.I = None
      self.name = None
      self.density = None

  def material_from_mass_ratios(self, Zs: list[float], As: list[float], mass_ratios: list[float]):
    mass_ratios = np.array(mass_ratios) / np.sum(mass_ratios)  # renormalise

    # calculate weighted average Z and electron density (very rough estimate!)
    n = 0
    weighted_Z = 0

    for i, Z in enumerate(Zs):
      n += electron_number_density(Z, self.density, As[i]) * mass_ratios[i]
      weighted_Z += Z * mass_ratios[i]

    self.n = n
    self.I = 10 * e * weighted_Z

  def material_from_Z_A(self, Z: int, A: float):
    self.n = electron_number_density(Z, self.density, A)
    self.I = 10 * e * Z