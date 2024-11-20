import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import get_cmap
import seaborn as sns

from EnergyCalibrationPy.utils import calculate_resolution_and_fwhm


def plot_calibrated_spectrum(data,
                             energy_values,
                             values,
                             x_variable,
                             x_limits=None,
                             y_limits=None,
                             color='blue',
                             title='Energy Calibration',
                             x_label='Energy [keV]',
                             y_label='Counts',
                             figsize=(12, 8),
                             save_plot=False,
                             save_file_name=None,
                             plot_dpi=None,
                             fontsize=16):
    """
    Plots the calibrated energy spectrum with user-defined settings.
    
    Parameters:
        data (pd.DataFrame): Data containing 'x' and 'counts' columns.
        energy_values (list or array): Known energy values for calibration.
        values (list or array): Corresponding area values for calibration.
        x_limits (tuple or None): Tuple defining x-axis limits (xmin, xmax). Default is None.
        color (str): Color of the calibrated spectrum plot.
        title (str): Title of the plot.
        x_label (str): X-axis label.
        y_label (str): Y-axis label.
        figsize (tuple): Size of the figure. Default is (12, 8).
    """
    if len(energy_values) != len(values):
        raise ValueError("Energy values and area values must have the same length.")

    # Perform linear calibration (regression)
    model = np.polyfit(values, energy_values, 1)
    calibrated_energy = np.polyval(model, data[x_variable].to_numpy())

    # Plot the calibrated spectrum
    plt.figure(figsize=figsize)
    plt.plot(calibrated_energy, data['counts'], color=color, label='Calibrated Spectrum')

    # Drop vertical lines at identified peaks with automatic colors
    cmap = get_cmap("tab10")  # Use a color map for automatic coloring
    for i, energy in enumerate(energy_values):
        plt.axvline(energy, color=cmap(i % 10), linestyle='--', label=f'{energy} keV')

    # Apply user-specified x-axis limits
    if x_limits:
        plt.xlim(x_limits)

    # Labels and title
    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.legend(loc='upper right', fontsize=fontsize)
    plt.tight_layout()
    plt.grid(False)
    if save_plot:
        plt.savefig(save_file_name, dpi=plot_dpi)
    plt.show()


######################################################################################################

def process_and_plot_files(file_path, x_variable, y_variable, x_label, y_label, legend_label, plot_color, fontsize=16,
                           save_file_name=None, save_plot=None, plot_dpi=None):
    filename = file_path.split('/')[-1]
    with open(file_path, 'r') as file:
        first_line = file.readline()
        if first_line.startswith("#"):
            df = pd.read_csv(file_path, skiprows=1)
        else:
            df = pd.read_csv(file_path)
    df.columns = [x_variable, y_variable]
    plt.figure(figsize=(10, 6))  # Adjust figure size here
    plt.plot(df[x_variable], df[y_variable], color=plot_color, label = legend_label)
    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.title(f'Plot of {filename}', fontsize=fontsize)
    plt.legend(loc='best', fontsize=fontsize - 2)
    plt.tight_layout()
    plt.grid(False)
    if save_plot:
        plt.savefig(save_file_name, dpi=plot_dpi)
    plt.show()


######################################################################################################

def regression_and_plot_with_peaks(energy_values, area_values, title='Regression Plot of Energy vs. Area',
                                   plot_color='red',
                                   co60=False, cs137=False, na22=False, ba133=False,
                                   extrapolation_range=(10, 1500), num_points=500, x_label=None, fontsize=16,
                                   y_label=None, save_plot=None, save_file_name=None, plot_dpi=None):
    """
    Performs linear regression on energy-area data, plots the data points, regression line, and confidence interval,
    and highlights peaks for specific isotopes.

    Parameters:
        energy_values (list or array): Known energy values.
        area_values (list or array): Corresponding area values.
        title (str): Title of the plot.
        plot_color (str): Color of the regression line and confidence interval.
        co60_peaks (list): Energies for Co-60 peaks.
        cs137_peak (float): Energy for Cs-137 peak.
        na22_peaks (list): Energies for Na-22 peaks.
        ba133_peaks (list): Energies for Ba-133 peaks.
        extrapolation_range (tuple): Range for extrapolation in energy values.
        num_points (int): Number of points for extrapolated line.
    """
    # Convert energy and area to numpy arrays
    energy = np.array(energy_values)
    area = np.array(area_values)

    # Perform linear regression
    polynomial_order = 1
    coefficients = np.polyfit(energy, area, polynomial_order)
    model = np.poly1d(coefficients)

    # Define extended energy range for extrapolation
    x_range = np.linspace(extrapolation_range[0], extrapolation_range[1], num_points)
    area_extrapolated = model(x_range)

    # Calculate residuals and standard deviation for confidence interval
    y_predicted = model(energy)
    residuals = area - y_predicted
    std_dev = np.std(residuals)

    # Plot original data points, regression line, and confidence interval
    plt.figure(figsize=(12, 8))
    plt.scatter(energy, area, color='blue', label='Data points')
    plt.plot(x_range, area_extrapolated, color=plot_color, linewidth=2, label='Extrapolated Regression Line')
    plt.fill_between(x_range, area_extrapolated - std_dev, area_extrapolated + std_dev, alpha=0.25, color=plot_color)

    # Annotate each data point with its energy value
    for x, y in zip(energy, area):
        plt.text(x, y, f'{x:.2f}', fontsize=10, ha='right')

    # Display Co-60 peaks with (energy, area) annotation in black
    if co60:
        co60_peaks = [1173.2, 1332.5]
        for peak_energy in co60_peaks:
            peak_area = model(peak_energy)
            plt.plot(peak_energy, peak_area, 'o', color='black',
                     label='Co-60 Peaks' if peak_energy == co60_peaks[0] else "")
            plt.text(peak_energy, peak_area, f'({peak_energy:.1f} keV, {peak_area:.2e} Wb)', color='black', fontsize=10,
                     ha='left')

    # Display Cs-137 peak with (energy, area) annotation in maroon
    if cs137:
        cs137_peak = 662
        cs137_area = model(cs137_peak)
        plt.plot(cs137_peak, cs137_area, 'o', color='maroon', label='Cs-137 Peak')
        plt.text(cs137_peak, cs137_area, f'({cs137_peak} keV, {cs137_area:.2e} Wb)', color='maroon', fontsize=10,
                 ha='left')

    # Display Na-22 peaks with (energy, area) annotation in cyan
    if na22:
        na22_peaks = [511, 1274]  # Energies for Na-22 in keV
        for peak_energy in na22_peaks:
            peak_area = model(peak_energy)
            plt.plot(peak_energy, peak_area, 'o', color='cyan',
                     label='Na-22 Peaks' if peak_energy == na22_peaks[0] else "")
            plt.text(peak_energy, peak_area, f'({peak_energy:.1f} keV, {peak_area:.2e} Wb)', color='cyan', fontsize=10,
                     ha='left')

    # Display Ba-133 peaks with (energy, area) annotation in yellow
    if ba133:
        ba133_peaks = [81, 356]  # Energies for Ba-133 in keV
        for peak_energy in ba133_peaks:
            peak_area = model(peak_energy)
            plt.plot(peak_energy, peak_area, 'o', color='magenta',
                     label='Ba-133 Peaks' if peak_energy == ba133_peaks[0] else "")
            plt.text(peak_energy, peak_area, f'({peak_energy:.1f} keV, {peak_area:.2e} Wb)', color='magenta',
                     fontsize=10, ha='left')

    # Labeling axes and title
    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.legend(loc='best', fontsize=fontsize - 2)
    plt.tight_layout()
    plt.grid(False)
    if save_plot:
        plt.savefig(save_file_name, dpi=plot_dpi)
    plt.show()


######################################################################################################


def plot_fwhm_resolution(mu_values, sigma_values, x_label=None, fontsize=16, save_plot=None, save_file_name=None,
                         plot_dpi=None, figure_size = None):
    """
    Creates regression plots for FWHM and Resolution against Mu values.

    Parameters
    ----------
    mu_values : array-like
        Array or list containing Mu values in terms of area (e.g., [Wb]).
    sigma_values : array-like
        Array or list containing standard deviation values corresponding to the Mu values.

    Returns
    -------
    None
        Displays the generated regression plots. Saves the figure if a file path is provided.
    """
    # Create a DataFrame for plotting
    fwhm, res = calculate_resolution_and_fwhm(sigma_values, mu_values)

    data = pd.DataFrame({'Mu': mu_values,
                         'FWHM': fwhm,
                         'Resolution': res})

    # Create the regression plots
    f, ax = plt.subplots(1, 2, figsize=figure_size)

    # First plot: Mu vs FWHM
    sns.regplot(x='Mu', y='FWHM', data=data, ci=None, scatter_kws={'s': 100}, line_kws={'color': 'red'}, ax=ax[0])
    ax[0].set_xlabel(x_label, fontsize=fontsize)
    ax[0].set_ylabel('FWHM', fontsize=fontsize)
    ax[0].set_title('Full Width Half Maximum', fontsize=fontsize)
    ax[0].grid(False)

    # Second plot: Mu vs Resolution
    sns.regplot(x='Mu', y='Resolution', data=data, ci=None, scatter_kws={'s': 100}, line_kws={'color': 'green'},
                ax=ax[1])
    ax[1].set_xlabel(x_label, fontsize=fontsize)
    ax[1].set_ylabel('Resolution', fontsize=fontsize)
    ax[1].set_title('Energy Resolution %', fontsize=fontsize)
    ax[1].grid(False)
    f.tight_layout()

    # Save the figure if a save path is provided
    if save_plot:
        plt.savefig(save_file_name, dpi=plot_dpi)
    plt.show()
