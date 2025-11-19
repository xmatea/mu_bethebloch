import numpy as np
import scipy
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from abc import ABC

from scipy.optimize import curve_fit

import g4_sim
from g4_sim import gaussian

rng = np.random.default_rng()

colours = np.array([(0, 0, 0, 0.8), (0, 0, 0, 0.6), (0, 0, 0, 0.5), (0, 0, 0, 0.4), (0, 0, 0, 0.6)])
default_cmap = LinearSegmentedColormap.from_list("greyscale", colours, N=len(colours))

colours2 = np.array([(1, 1, 1, 0), (222/255, 137/255, 190/255, 1)])
single_mask_cmap = LinearSegmentedColormap.from_list("single_mask_cmap", colours2, N=len(colours))

# universal constants (should do this in nat units really!)
e_0 = 8.85418 * 1e-12
m_e = 9.10938 * 1e-31
c = 2.99892 * 1e8
N_A = 6.02214 * 1e23
M_u = 1
e = 1.60218 * 1e-19
pi = np.pi

m_mu = 1.8835 * 1e-28


def electron_number_density(Z: int, rho: float, A: int) -> float:
    """
    Calculated electron number density from Z, A and density
    Args:
        Z: Atomic number
        rho: density (g/cm^3)
        A: Atomic weight (amu)

    Returns:
        Electron number density (n/m^3)
    """
    return ((N_A * Z * rho) / (A * M_u)) * 1e6


def p_to_E(p: float, m: float) -> float:
    """
    Calculates total kinetic energy of a particle from lab frame momentum
    Args:
        p: Momentum (kg*m/s)
        m: particle mass (kg)

    Returns:
        Total relativistic energy (J)
    """
    return np.sqrt((p * c) ** 2 + (m * c ** 2) ** 2)

def beta2_to_E(beta2: float, m: float) -> float:
    """
    Calculates total relativistic energy of a particle from beta squared

    Args:
        beta2: beta squared - (v/c)**2
        m: particle mass (kg)

    Returns:
        Total relativistic energy (J)
    """
    return m * c * c * np.sqrt(1 + beta2)


def E_to_beta2(E: float, m: float) -> float:
    """
    Calculates beta squared from total relativistic energy of a particle

    Args:
        E: Total relativistic energy (J)
        m: Particle mass (kg)

    Returns:
        beta squared (v/c)**2
    """

    return (E * E) / (m * c * c) ** 2 - 1


def bethe_bloch(n: float, I: float, z: int, beta2: float) -> float:
    """
    Calculates rate of energy loss dE/dx given beta squared for a heavy charged particle in matter.

    Args:
        n: electron number density ()
        I: mean excitation energy (J)
        z: charge multiple of electron charge
        beta2: beta squared (v/c)**2

    Returns:
        Rate of energy loss dE/dx (J/m)
    """
    c1 = 4 * pi / (m_e * c ** 2)
    c2 = (e ** 2 / (4 * pi * e_0)) ** 2
    c3 = 2 * m_e * c ** 2
    c4 = n * z * z

    return - (c1 * c2 * c4 / beta2) * np.log((c3 * beta2) / (I * (1 - beta2))) + c1 * c2 * c4

class Beam(ABC):
    def sample_muons(self):
        ...

class SampledBeam2D(Beam):
    def __init__(self, line: np.ndarray, width: float, max_momentum: float, min_momentum: float):
        """
        Creates a beam from arbitrary function.

        Args:
            line: function to sample momenta from (beam shape)
            width: width of beam (mm)
            max_momentum: maximum momentum of beam signal
            min_momentum: minimum momentum of beam signal
        """
        super().__init__()

        self.width = width
        self.line = line
        self.max_momentum = max_momentum
        self.min_momentum = min_momentum

    def sample_muons(self, n: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Samples muons from the beam
        Args:
            n: number of muons to sample (optional)

        Returns:
            muon positions centred around 0 (mm), momentum values, angles (angles not implemented yet)
        """

        line = (self.line - np.min(self.line))

        line = line / np.max(line)

        dp = self.max_momentum-self.min_momentum

        momenta = self.min_momentum + line*dp

        pos = np.linspace(-self.width/2, self.width/2, len(line))
        angle = 0

        plt.figure()
        plt.title("Beam function")
        plt.plot(pos, momenta)
        plt.xlabel("Position (mm)")
        plt.ylabel("Muon momentum (MeV/c)")
        plt.show()

        return pos, momenta, angle


class GaussianBeam2D(Beam):
    def __init__(self, sigma: float, momentum: float, momentum_spread: float, n):
        """
        Initialise beam object.

        Args:
            sigma: beam sigma (mm)
            momentum: momentum value (MeV/c)
            momentum_spread: momentum spread standard deviation (%)
        """

        self.sigma = sigma
        self.momentum = momentum
        self.momentum_spread = momentum_spread / 100
        self.angle_dist = 0
        self.n = n

    def sample_muons(self, n: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Samples muons from the beam
        Args:
            n: number of muons to sample (optional)

        Returns:
            muon positions centred around 0 (mm), momentum values, angles (angles not implemented yet)
        """
        if n is None:
            n = self.n

        pos = rng.normal(loc=0, scale=self.sigma, size=n)
        angle = None

        mom_sigma = self.momentum_spread*self.momentum
        momentum = rng.normal(loc=self.momentum, scale=mom_sigma, size=n)

        return pos, momentum, angle

class Sample2D:
    def __init__(self, mask: np.ndarray, width: float, height: float):
        """
        Creates a new sample.

        Args:
            mask: image (np array)
            width: image width
            height: image height
        """
        self.shape_mask = mask

        self.width = width
        self.height = height

        self.n_materials = np.unique(self.shape_mask) # get all unique values in shape mask

        self.material_mask = np.zeros((*self.shape_mask.shape, 3))

        self.material_map = None

    def set_materials(self, material_maps: list[tuple]):
        """
        Maps materials to colour levels.
        Args:
            material_maps: List of tuples, with the form (colour value, material object)

        Returns:
            None
        """
        self.material_map = material_maps
        self.material_mask[:, :, 0] = self.shape_mask

        for value, material in material_maps:
            self.material_mask[self.shape_mask == value, 1] = material.n
            self.material_mask[self.shape_mask == value, 2] = material.I


class Material:
    def __init__(self, name: str, density: float):
        """
        Creates a new material.
        Args:
            name: material name
            density: material density (g/cm^3)
        """
        self.name = name
        self.density = density

        self.n = None
        self.I = None

    def material_from_mass_ratios(self, Zs: list[float], As: list[float], mass_ratios: list[float]):
        """
        Calculates electron density n and mean exitation energy I based on list of Z, list of A and mass ratios of each
        element
        Args:
            Zs: list of atomic numbers
            As: list of atomic weights
            mass_ratios: list of mass ratios

        Returns:
            None
        """
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
        """
        Creates a new material based on Z and A for a single element
        Args:
            Z: Atomic number
            A: Atomic weight

        Returns:
            None
        """

        self.n = electron_number_density(Z, self.density, A)
        self.I = 10 * e * Z


class BBsimulation2D:
    def __init__(self, beam: Beam, sample: Sample2D, sample_pos: float = 0, world_material: Material = None, cmap=None):
        """
        Simulator object responsible for running 2D simulations.

        Args:
          beam: Beam object
          sample: Sample object to simulate for
          sample_pos: Distance between edge of sample and beam window
          world_material: Surrounding material. Default is vaccuum.
        """
        self.beam = beam
        self.sample = sample

        self.beampos = None

        self.world_material = world_material

        if cmap is None:
            self.cmap = default_cmap
        else:
            self.cmap = cmap

    def run(self, plot_res: bool=True):
        beam_pos = self.sample.width / 2 # position muon beam at top centre of image
        sample = self.sample

        positions, momenta, _ = self.beam.sample_muons()

        # pixel width of mask
        pix_w = sample.material_mask[:,:,0].shape[0]

        # calculate muon positions in index space
        norm_pos = (beam_pos + positions) / sample.width
        index_pos = np.clip(norm_pos * pix_w, a_min=0, a_max=pix_w-1).astype(int)

        # Calculate muon energies
        mom_SI = momenta * 1e6 * e / c
        E_0 = m_mu * c * c
        E = np.sqrt((mom_SI * c) ** 2 + (m_mu * c * c) ** 2)

        # Store kinetic energy for each muon everywhere
        E_k = np.zeros((len(index_pos), pix_w))

        dx = (sample.width / pix_w) / 1e3 # step size determined by image resolution

        for i in range(pix_w):
            # sample electron energies and mean exitation energies at current position for each muon
            n = sample.material_mask[i, index_pos, 1]
            I = sample.material_mask[i, index_pos, 2]

            # Calculate (v/c)^2 for each muon
            beta_E = E_to_beta2(E, m_mu)

            # Calculate energy loss and update energies
            E += bethe_bloch(n, I, z=1, beta2=beta_E) * dx
            E_k[:, i] = E - E_0

        E_k = np.nan_to_num(E_k, nan=0) # remove all nans
        E_k[E_k < 0] = 0 # remove any negatives (can happen due to weird floating point or nan stuff idk, i should use nat units)

        E_k = E_k / (1e6 * e)

        if plot_res:
            self.plot_res(E_k, index_pos, pix_w)

        return E_k, index_pos


    def plot_res(self, E_k, muon_pos, img_width_px):
        # decay positions will be the first point kinetic energy reaches 0
        d_pos = np.argmin(E_k, axis = 1)

        # create image to show decay positions
        decay_pos = np.zeros_like(self.sample.shape_mask)
        decay_pos[d_pos, muon_pos] = 1

        # create image to show beam trajectories
        sample_trajectories = Image.fromarray(np.ones_like(self.sample.shape_mask))
        draw = ImageDraw.Draw(sample_trajectories)

        for i, pos in enumerate(muon_pos[:999]):
            draw.line([(pos, 0), (pos, d_pos[i])], fill=3)

        # calculate the extent for images
        real_x = np.linspace(0, self.sample.width, img_width_px)
        real_y = np.linspace(0, self.sample.height, img_width_px)
        dx = (real_x[1] - real_x[0]) / 2.
        dy = (real_y[1] - real_y[0]) / 2.

        extent = [real_x[0] - dx, real_x[-1] + dx, real_y[0] - dy, real_y[-1] + dy]

        # plotting
        fig, ax = plt.subplots(ncols=2)

        ax[0].set_title("Sample trajectories for 1000 muons")
        ax[0].imshow(self.sample.shape_mask, cmap=self.cmap, extent=extent)
        ax[0].imshow(sample_trajectories, cmap=single_mask_cmap, extent=extent)

        ax[1].set_title("Full implantation profile")
        ax[1].imshow(self.sample.shape_mask, cmap=self.cmap, extent=extent)
        ax[1].imshow(decay_pos, cmap=single_mask_cmap, extent=extent)


        ax[0].set_xlabel("mm")
        ax[1].set_xlabel("mm")

        plt.show()

def hello_kitty_sim():
    im = Image.open("hello_kitty.png").convert("L")

    img = np.array(im)

    # set colour levels to 0, 1, 2, 3, 4
    for i, val in enumerate(np.unique(img)):
        img[img == val] = i

    mask = np.array(img)

    # create sample and materals
    sample = Sample2D(mask, 10, 10)

    al_mat = Material("Al", density=2.7)
    al_mat.material_from_Z_A(Z=13, A=27)

    ruby = Material("AlOx", density=3.98)
    ruby.material_from_mass_ratios([13, 8], [27, 16], [27 * 2, 16 * 3])

    water = Material("H2O", density=1)
    water.material_from_mass_ratios([1, 8], [1, 16], [1, 16 * 2])

    fe_mat = Material("Fe", density=7.874)
    fe_mat.material_from_Z_A(Z=26, A=55.84)

    gold = Material("Au", density=7.874)
    gold.material_from_Z_A(Z=26, A=55.84)

    air = Material("Air", density=1.2 / 1e3)
    air.material_from_mass_ratios(Zs=(7, 8, 18), As=(14, 16, 40), mass_ratios=(0.755, 0.2314, 0.0129))

    # map colours in image to materials
    sample.set_materials(((0, al_mat), (1, ruby), (2, gold), (3, water), (4, air)))

    # create colour map for display
    hk_colours = np.array(
        [[0, 0, 0], [90.6, 13.3, 20.8], [93.3, 78.4, 15.3], [88.8, 100.0, 100], [100, 100, 100]]) / 100
    cmap = LinearSegmentedColormap.from_list("hk_cmap", hk_colours, N=len(colours))

    beam = GaussianBeam2D(sigma=1.5, momentum=40, momentum_spread=4, n=100_000)
    simulation = BBsimulation2D(beam, sample, sample_pos=0, cmap=cmap)

    simulation.run()

def fe_al_sim():
    im = Image.new(mode="L", size=(500,500))

    draw = ImageDraw.Draw(im)

    # draw two rectangles with values 0 and 1
    draw.rectangle(((0,0), (500, 250)), fill=0)
    draw.rectangle(((0, 250), (500, 500)), fill=1)

    mask = np.array(im)
    sample = Sample2D(mask, 4, 4)

    # create materials
    al = Material("Al", density=2.7)
    al.material_from_Z_A(Z=13, A=27)

    fe = Material("Fe", density=7.874)
    fe.material_from_Z_A(Z=26, A=55.84)

    sample.set_materials(((0, al), (1, fe)))

    # create colour map
    colours = np.array([(0,0,0,0.4), (0,0,0,0.7)])
    cmap = LinearSegmentedColormap.from_list("grey_cmap", colours, N=len(colours))

    beam = GaussianBeam2D(sigma=1.5, momentum=50, momentum_spread=4, n=100_000)

    simulation = BBsimulation2D(beam, sample, sample_pos=0, cmap=cmap)
    E_s, _ = simulation.run()

def reconstruction_tests():
    im = Image.new(mode="L", size=(500, 500))

    draw = ImageDraw.Draw(im)

    ratio = 5.3 / 50
    px_radius = int(ratio * 500)

    draw.circle((250, 0), radius=px_radius, fill=1)
    draw.circle((250, px_radius * 2), radius=px_radius, fill=2)
    # draw.circle((250, 250), radius=100, fill=2)

    mask = np.array(im)
    sample = Sample2D(mask, 50, 50)

    al = Material("Al", density=2.7)
    al.material_from_Z_A(Z=13, A=27)

    fe = Material("Fe", density=7.874)
    fe.material_from_Z_A(Z=26, A=55.84)

    air = Material("Air", density=1.2 / 1e3)
    air.material_from_mass_ratios(Zs=(7, 8, 18), As=(14, 16, 40), mass_ratios=(0.755, 0.2314, 0.0129))

    sample.set_materials([(1, air), (0, al), (2, al)])
    # colours = np.array([(0, 0, 0, 0.4)])
    # cmap = LinearSegmentedColormap.from_list("cmap", colours, N=len(colours))

    width = 50  # in mm
    img = g4_sim.load_image("battery-65.txt", width=250)

    line = img[140, :]

    # line = np.sin(np.arange(0, 500)/(6*np.pi))

    beam = SampledBeam2D(line=line, width=width, min_momentum=50, max_momentum=65)

    simulation = BBsimulation2D(beam, sample, sample_pos=0)

    E_k, muon_pos, = simulation.run()

def triangle_rot():
    al = Material("Al", density=2.7)
    al.material_from_Z_A(Z=13, A=27)

    air = Material("Air", density=1.2 / 1e3)
    air.material_from_mass_ratios(Zs=(7, 8, 18), As=(14, 16, 40), mass_ratios=(0.755, 0.2314, 0.0129))

    im = Image.new(mode="L", size=(500, 500))

    mask = np.array(im)

    sample = Sample2D(mask, 10, 10)

    sample.set_materials([(0, air)])
    colours = np.array([(0, 0, 0, 0.4), (0, 0, 0, 0.1)])
    cmap = LinearSegmentedColormap.from_list("cmap", colours, N=len(colours))

    beam = GaussianBeam2D(sigma=1.5, momentum=40, momentum_spread=4, n=100_000)

    simulation = BBsimulation2D(beam, sample, sample_pos=0, cmap=cmap)
    E_k, muon_pos, = simulation.run(plot_res=False)

    E_f = E_k[:, -1]
    arrived_muons = muon_pos[E_f > 0]
    vals, counts = np.unique_counts(arrived_muons)

    popt, _ = curve_fit(gaussian, vals, counts, p0=(250, 100, 100_000))

    angles = np.arange(0, 120, 30)

    for theta in angles:
        print(f"calculating theta = {theta:.3f}")
        im = Image.new(mode="L", size=(500, 500))

        draw = ImageDraw.Draw(im)
        draw.regular_polygon((250, 250, 150), 3, fill=1)

        mask = np.array(im)

        mask = scipy.ndimage.rotate(mask, theta)

        sample = Sample2D(mask, 10, 10)

        sample.set_materials([(0, air), (1, al)])
        colours = np.array([(0, 0, 0, 0.4), (0, 0, 0, 0.1)])
        cmap = LinearSegmentedColormap.from_list("cmap", colours, N=len(colours))

        beam = GaussianBeam2D(sigma=1.5, momentum=40, momentum_spread=4, n=100_000)

        simulation = BBsimulation2D(beam, sample, sample_pos=0, cmap=cmap, world_material=None)
        E_k, muon_pos, = simulation.run(plot_res=False)

        E_f = E_k[:, -1]
        arrived_muons = muon_pos[E_f > 0]
        vals, counts = np.unique_counts(arrived_muons)

        norm_counts = counts - gaussian(vals, *popt)

        # plotting
        img_width_px = 500

        # decay positions will be the first point kinetic energy reaches 0
        d_pos = np.argmin(E_k, axis = 1)

        # create image to show decay positions
        decay_pos = np.zeros_like(sample.shape_mask)
        decay_pos[d_pos, muon_pos] = 1

        # create image to show beam trajectories
        sample_trajectories = Image.fromarray(np.ones_like(sample.shape_mask))
        draw = ImageDraw.Draw(sample_trajectories)

        for i, pos in enumerate(muon_pos[:999]):
            draw.line([(pos, 0), (pos, d_pos[i])], fill=3)

        # calculate the extent for images
        real_x = np.linspace(0, sample.width, img_width_px)
        real_y = np.linspace(0, sample.height, img_width_px)
        dx = (real_x[1] - real_x[0]) / 2.
        dy = (real_y[1] - real_y[0]) / 2.

        extent = [real_x[0] - dx, real_x[-1] + dx, real_y[0] - dy, real_y[-1] + dy]

        # plotting
        fig, ax = plt.subplots(ncols=2)

        ax[0].set_title("Sample trajectories for 1000 muons")
        ax[0].imshow(sample.shape_mask, cmap=cmap, extent=extent)
        ax[0].imshow(sample_trajectories, cmap=single_mask_cmap, extent=extent)

        ax[1].set_title(f"Distribution at detector")
        ax[1].plot(vals, counts, label="uncorrected")
        #ax[1].plot(vals, norm_counts, label="corrected")
        #ax[1].plot(vals, gaussian(vals, *popt))
        ax[1].legend()

        fig.suptitle(f"Theta = {theta:.0f}")
        plt.show()

if __name__ == "__main__":
    #hello_kitty_sim()
    #fe_al_sim()
    reconstruction_tests()
    #triangle_rot()
