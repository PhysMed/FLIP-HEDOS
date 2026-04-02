# Under the GNU General Public License v3.0 (GPLv3):
# Copyright (C) 2026 PhysMed Research Group - University of Navarra
#
# This file includes code licensed under the MIT License:
# Copyright (c) 2021-2023 MGH Radiation Oncology
#
# ========= Updated by Chris Beekman et al. 2023 ===========
# ========= Modified by Marina Garcia-Cardosa since January 2024 to manage with data patients =========
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def plot_dose_distribution(blood_dose_total, dose_contributions, mean_blood_dose=None):
    x_max1 = 2 * np.percentile(blood_dose_total.dose, 90)
    x_max2 = max([2 * np.percentile(dose, 90) for dose in dose_contributions.values()])
    bins1 = np.linspace(0, x_max1, 100)
    bins2 = np.linspace(0, x_max2, 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hist(blood_dose_total.dose, bins=bins1)
    ax1.axvline(np.mean(blood_dose_total.dose), ymax=1, c='k', linestyle='-',
                label='Simulated mean dose - {:.3f} Gy'.format(np.mean(blood_dose_total.dose)))
    if mean_blood_dose is not None:
        ax1.axvline(mean_blood_dose, ymax=1, c='red', linestyle='--',
                    label='Expected mean dose - {:.3f} Gy'.format(mean_blood_dose))
    for organ, dose in dose_contributions.items():
        # added by Marina to always the same colors
        line_color = choose_color_plot(organ)
        ax2.hist(dose, bins=bins2[1:], histtype='step', linewidth=1.5, alpha=0.7, label=organ, color=line_color)

    ax1.set_xlabel('Dose (Gy)')
    ax2.set_xlabel('Dose (Gy)')
    ax1.set_title('Blood dose histogram - All fractions')
    ax2.set_title('Blood dose contributions - One fraction')
    ax1.legend()
    ax2.legend()

    plt.show()


def calculate_dvh_patient_specific(dose_contributions, num_particles, organ):
    dvh_values, bins = np.histogram(dose_contributions[organ], bins=num_particles, density=True)
    dvh_values = np.cumsum(dvh_values[::-1])[::-1] / np.sum(dvh_values)
    plot_dvh(dvh_values, bins, organ)


def plot_dvh(dvh_values, bins, organ):
    line_color = choose_color_plot(organ)
    plt.plot(bins[:-1], dvh_values * 100, linestyle='-', linewidth=1.5, label=organ, color=line_color)
    plt.title('Dose volume histograms - One fraction')
    plt.xlabel('Dose (Gy)', fontsize=14)
    plt.ylabel('Blood volume (%)', fontsize=14)
    plt.ylim(0, 20)
    plt.xlim(-0.01, 1)
    plt.legend()
    plt.grid(True)
    plt.show(block=False)


def choose_color_plot(organ):
    """
        A predefined color string for plotting based on the given organ name.
        - Parameters:
            organ (str): Name of the organ or vascular structure.
        - Returns:
            str or None: A color name recognized by plotting libraries (e.g., matplotlib),
                         or None if the organ is not recognized.
    """
    if organ == 'left_heart':
        line_color = 'navy'
    elif organ == 'right_heart':
        line_color = 'darkred'
    elif organ == 'aorta':
        line_color = 'salmon'
    elif organ == 'inferior_vena_cava':
        line_color = 'skyblue'
    elif organ == 'liver':
        line_color = 'gray'
    elif organ == 'kidney':
        line_color = 'gold'
    elif organ == 'stomach_oesophagus':
        line_color = 'darkorange'
    elif organ == 'spleen':
        line_color = 'violet'
    elif organ == 'pancreas':
        line_color = 'green'
    elif organ == 'flip_arterial':
        line_color = 'red'
    elif organ == 'flip_venous':
        line_color = 'blue'
    elif organ == 'specific_vasculature':
        line_color = 'red'
    else:
        line_color = None
    return line_color


def plot_volumes(volume_ref, volume, plot_slice=None, cmap_ref='Greys_r', cmap='Greys_r', scrollable=False):
    """
    Plotting method to visualize 3D volumes.
    Either visualize a slice, or scroll through the entire volume in interactive mode.
    """
    if plot_slice is None:
        plot_slice = np.argmax(np.sum(volume, axis=(0, 1)))

    if scrollable:
        backend = matplotlib.get_backend()
        # you need interactive mode for scrolling, on Mac this works:
        matplotlib.use("QtAgg")
        fig, ax = plt.subplots(1, 1)
        tracker = IndexTracker(ax, volume_ref, volume, plot_slice, cmap_ref, cmap)
        fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
        plt.show()
        # return to original backend.
        matplotlib.use(backend)
    else:
        fig, ax = plt.subplots(1, 1)
        img = ax.imshow(volume_ref[:, :, plot_slice], cmap=cmap_ref)
        c_bar = fig.colorbar(img)
        c_bar.set_label('Treatment dose at slice {} (Gy)'.format(plot_slice))
        ax.imshow(volume[:, :, plot_slice], cmap=cmap, alpha=0.75)
        plt.show()


class IndexTracker(object):
    def __init__(self, ax, X, Y, plot_slice, cmap_ref='Greys', cmap='Greys_r'):
        self.ax = ax
        self.X = X
        self.Y = Y
        self.plot_slice = plot_slice
        _, _, self.slices = X.shape

        self.im1 = ax.imshow(self.X[:, :, self.plot_slice], cmap=cmap_ref)
        self.im2 = ax.imshow(self.Y[:, :, self.plot_slice], cmap=cmap, alpha=0.75)

        c_bar = plt.colorbar(self.im1)
        c_bar.set_label('Treatment dose (Gy)')

        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.plot_slice = (self.plot_slice + 1) % self.slices
        else:
            self.plot_slice = (self.plot_slice - 1) % self.slices
        self.update()

    def update(self):
        im1_data = self.im1.to_rgba(self.X[:, :, self.plot_slice], alpha=self.im1.get_alpha())
        im2_data = self.im2.to_rgba(self.Y[:, :, self.plot_slice], alpha=self.im2.get_alpha())

        self.im1.set_data(im1_data)
        self.im2.set_data(im2_data)

        self.ax.set_ylabel('slice %s' % self.plot_slice)
        self.im1.axes.figure.canvas.draw()
        self.im2.axes.figure.canvas.draw()
