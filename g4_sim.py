import subprocess
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

def load_image(path: str, plane: tuple[float] = (0,1), width: int = 200):
    data = np.loadtxt(path, skiprows=3, delimiter=" ")

    # get locations of tracked events
    xdata, ydata = data[:, plane[0]], data[:, plane[1]]

    # create empty image
    image = np.zeros((width,width))

    # translate x and y positions so that all values are positive (centered around 0 by default)
    xdata = xdata - np.min(xdata)
    ydata = ydata - np.min(ydata)

    # normalise and scale x and y coordinates to be between 0 and width-1
    # so that they can represent indices /pixels in the image
    xdata = np.round(xdata / np.max(xdata) * (width-1))
    ydata = np.round(ydata / np.max(ydata) * (width-1))

    # loop through each event and count up the pixels
    for i, _ in enumerate(xdata):
        x = int(xdata[i])
        y = int(ydata[i])
        image[y, x] += 1

    return image

def load_beamloss(path, vox=1, xlim=(-10, 10), ylim=(-10, 10), zlim=(0, 250)):
    data = np.loadtxt(path, skiprows=3, delimiter=" ")

    # get locations of tracked events
    xdata_r, ydata_r, zdata_r = data[:, 0], data[:, 1], data[:, 2]

    xdata = xdata_r[(xdata_r > xlim[0]) & (xdata_r < xlim[1]) & (ydata_r > ylim[0]) & (ydata_r < ylim[1]) & (zdata_r > zlim[0]) & (zdata_r < zlim[1])]
    ydata = ydata_r[(xdata_r > xlim[0]) & (xdata_r < xlim[1]) & (ydata_r > ylim[0]) & (ydata_r < ylim[1]) & (zdata_r > zlim[0]) & (zdata_r < zlim[1])]
    zdata = zdata_r[(xdata_r > xlim[0]) & (xdata_r < xlim[1]) & (ydata_r > ylim[0]) & (ydata_r < ylim[1]) & (zdata_r > zlim[0]) & (zdata_r < zlim[1])]

    n_x = int((np.max(xdata) - np.min(xdata))//vox)
    n_y = int((np.max(ydata) - np.min(ydata))//vox)
    n_z = int((np.max(zdata) - np.min(zdata))//vox)

    # create empty image
    image = np.zeros((n_x, n_y, n_z))

    # translate x and y positions so that all values are positive (centered around 0 by default)
    xdata = xdata - np.min(xdata)
    ydata = ydata - np.min(ydata)
    zdata = zdata - np.min(zdata)

    # normalise and scale x and y coordinates to be between 0 and width-1
    # so that they can represent indices /pixels in the image
    xdata = np.round(xdata / np.max(xdata) * (n_x - 1))
    ydata = np.round(ydata / np.max(ydata) * (n_y - 1))
    zdata = np.round(zdata / np.max(zdata) * (n_z - 1))

    # loop through each event and count up the pixels
    for i, _ in enumerate(xdata):
        x = int(xdata[i])
        y = int(ydata[i])
        z = int(zdata[i])
        image[x, y, z] += 1

    #yslice = np.sum(image, axis=(0,1))

    plt.figure()
    plt.imshow(image[:, n_y//2, :])
    #plt.plot(np.linspace(0-150-0.05, len(yslice)*vox-150-0.05, len(yslice)), yslice)
    plt.xlim(xmin=0)
    plt.show()

    print(image.shape)

def gaussian(x, mean, sigma, a):
    return a*np.exp(-0.5*(x-mean)**2/sigma**2)

def load_beamloss_1D(path, mean=0.2, zlim=(0, 250)):
    data = np.loadtxt(path, skiprows=3, delimiter=" ")

    # get locations of tracked events
    zdata_r =  data[:, 2]
    zdata = zdata_r[(zdata_r > zlim[0]) & (zdata_r < zlim[1])] - 150 - 0.05

    values, counts = np.unique_counts(zdata)

    popt, pcov = curve_fit(gaussian, values, counts, p0=[mean, 5, 1e6])

    print(f"mean stopping position {popt[0]}\t{abs(popt[1])}")

    plt.figure()
    plt.plot(values, counts)
    plt.plot(values, gaussian(values, *popt))
    plt.xlabel("Depth (mm)")
    plt.ylabel("Muon counts")
    plt.suptitle("G4Beamline simulation")
    plt.xlim(xmin=0, xmax=100)
    plt.show()

def run_sim(momenta, filepath, elements, title):

    script = "\n".join([f"g4bl {filepath} filename={title}_{elem} momentum={mom} element={elem}\n" for mom in momenta for elem in elements])

    scriptpath = f"C:/g4beamline_files/momentum_scan_{title}.bat"
    with open(scriptpath, "w") as wfile:
        wfile.write(script)

    result = subprocess.run([scriptpath], capture_output=True)
    print(result.stdout.decode())

def plot(title, load_dir=None, momentum=None):
    for mom in momentum:
        #filepath_slice = f"{title}-slice-{mom}.txt"
        filepath_det = f"{title}-{mom}.txt"

        if dir is not None:
            #filepath_slice = f"{load_dir}/{filepath_slice}"
            filepath_det = f"{load_dir}/{filepath_det}"

        #slice_img = load_image(filepath_slice, plane=(0, 2), width=50)
        slice_det = load_image(filepath_det, plane=(0, 1), width=150)

        fig, ax = plt.subplots(ncols=2)
        ax[0].set_title(f"Side slice {mom}MeV")
        #ax[0].imshow(slice_img)

        ax[1].set_title(f"Detector response {mom}MeV")
        ax[1].imshow(slice_det)

        plt.show()

def plot_slice(title, load_dir=None, momentum=None, slice_heights=None):
    for mom in momentum:
        #filepath_slice = f"{title}-slice-{mom}.txt"
        filepath_det = f"{title}-{mom}.txt"

        if dir is not None:
            filepath_det = f"{load_dir}/{filepath_det}"

        slice_det = load_image(filepath_det, plane=(0, 1), width=150)


        fig, ax = plt.subplots(ncols=2)
        fig.suptitle(f"Procell AAA at {mom} MeV")
        ax[1].set_title(f"Slices")
        ax[1].set_xlabel("Position (px)")
        ax[1].set_ylabel("Muon counts")


        lines = np.zeros((len(slice_heights), slice_det.shape[0]))
        for i, slice in slice_heights:
            lines[i, :] = slice_det[slice, :]
            ax[1].plot(np.arange(0, len(line)), line, label=f"y = {slice} px")

        ax[1].legend()

        ax[0].set_title(f"Detector response")
        ax[0].imshow(slice_det)

        plt.show()

def process_trim_res(path, mean):
    x, n = np.loadtxt(path, skiprows=5, unpack=True, delimiter=",")
    x -= 0.117

    popt, pcov = curve_fit(gaussian, x, n, [mean, 2, 1000])

    print(f"{popt[0]}\t{abs(popt[1])}")

    plt.figure()
    plt.plot(x, n)
    plt.plot(x, gaussian(x, *popt))
    plt.show()

    def compare_plots():
        data = np.loadtxt(f"{folder}/mu_depth_SRIM.csv", delimiter=",", skiprows=2)
        datag4 = np.loadtxt(f"{folder}/mu_depth_G4.csv", delimiter=",", skiprows=2)
        datab = np.loadtxt(f"{folder}/mu_depth_BB.csv", delimiter=",", skiprows=1)

        momenta = [20, 35, 50, 65, 80]

        plt.figure()
        plt.errorbar(data[:, 0], data[:, 1], yerr=data[:, 2], marker="*", linestyle="-", label="Li SRIM")
        plt.errorbar(data[:, 0], data[:, 3], yerr=data[:, 4], marker="+", linestyle="-", label="Al SRIM")
        plt.errorbar(data[:, 0], data[:, 5], yerr=data[:, 6], marker="v", linestyle="-", label="Fe SRIM")
        plt.errorbar(data[:, 0], data[:, 7], yerr=data[:, 8], marker="x", linestyle="-", label="Ag SRIM")
        plt.errorbar(data[:, 0], data[:, 9], yerr=data[:, 10], marker="o", linestyle="-", label="Pb SRIM")

        plt.errorbar(datag4[:, 0], datag4[:, 1], yerr=datag4[:, 2], marker="*", linestyle="--", label="Li  G4")
        plt.errorbar(datag4[:, 0], datag4[:, 3], yerr=datag4[:, 4], marker="+", linestyle="--", label="Al  G4")
        plt.errorbar(datag4[:, 0], datag4[:, 5], yerr=datag4[:, 6], marker="v", linestyle="--", label="Fe  G4")
        plt.errorbar(datag4[:, 0], datag4[:, 7], yerr=datag4[:, 8], marker="x", linestyle="--", label="Ag  G4")
        plt.errorbar(datag4[:, 0], datag4[:, 9], yerr=datag4[:, 10], marker="o", linestyle="--", label="Pb G4")

        plt.plot(datab[:, 0], datab[:, 1], marker="*", linestyle="-", label="Li  BB")
        plt.plot(datab[:, 0], datab[:, 2], marker="+", linestyle="-", label="Al  BB")
        plt.plot(datab[:, 0], datab[:, 3], marker="v", linestyle=":", label="Fe  BB")
        plt.plot(datab[:, 0], datab[:, 4], marker="x", linestyle=":", label="Ag  BB")
        plt.plot(datab[:, 0], datab[:, 5], marker="o", linestyle="-", label="Pb BB")

        # plt.yscale("log")
        plt.legend(loc="upper left")

        plt.ylabel("Stopping depth (mm)")
        plt.xlabel("Muon momentum (MeV/c)")
        plt.show()

if __name__ == "__main__":
    folder = "C:/g4beamline_files/old results"

    #momenta = np.arange(65, 90, 5)

    #run_sim([35, 65], filepath="C:/g4beamline_files/srimlike.g4bl", title="depth_test", elements=["Fe", "Ag", "Pb"])

    #plot("egg", momenta)

    #momenta = np.arange(45, 80, 2.5)

    #load_beamloss(f"{folder}/battery-beamloss-55.txt", vox=1, xlim=(0, 10), ylim=(0, 10), zlim=(0, 250))

    #process_trim_res(f"{folder}/srim_curves/Pb/65.00000MeVc_total_profile.dat", 4)

    #load_beamloss_1D(f"{folder}/depth_test_Pb-beamloss-65.txt", mean=3)
    #plot("battery", load_dir=folder, momentum=[65])

    #plot_slice("battery", load_dir=folder, momentum=[65], slice_heights=[80, 110, 140])

    #load_slice("C:/g4beamline_files/old results/battery-65meV.txt", slice_height=)


    #compare_plots()
