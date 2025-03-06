import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import get_cmap
from scipy.interpolate import interp1d

from EnergyCalibrationPy.utils import calculate_resolution_and_fwhm


def plot_calibrated_spectrum(data,
                             energy_values,
                             values,
                             x_variable,
                             x_limits=None,
                             color='blue',
                             title='Energy Calibration',
                             x_label='Energy [keV]',
                             y_label='Counts',
                             fig_size=(12, 8),
                             save_plot=False,
                             save_file_name=None,
                             plot_dpi=None,
                             font_size=16):
    """
    Plots the calibrated energy spectrum with user-defined settings.

    Parameters
    ----------
    data: pd.DataFrame
        Data containing 'x' and 'counts' columns.
    energy_values: Union[list, np.ndarray]
        Known energy values for the calibration.
    values: Union[list, np.ndarray]
        Corresponding are values for the calibration.
    x_variable: str
        The name of the variable to be plotted on x-axis.
    x_limits: tuple
        The x-axis limits.
    color: str, optional
        The color of the calibrated spectrum plot.
    title: str, optional
        The title for the calibrated spectrum plot.
    x_label: str, optional
        The x-label for the calibrated spectrum plot.
    y_label: str, optional
        The y-label for the calibrated spectrum plot.
    fig_size: tuple, optional
        The size of the figure. Default is (12, 8).
    save_plot: bool, optional
        Whether to save the plot or not. Default is False.
    save_file_name: str, optional
        The name of the file to save the plot as.
    plot_dpi: int, optional
        The dpi of the plot.
    font_size: int, optional
        The size of the font used for the plot. Default is 16.
    """
    values = values[:, 0]

    if len(energy_values) != len(values):
        raise ValueError("Energy values and area values must have the same length.")

    # Perform linear calibration (regression)
    model = np.polyfit(values, energy_values, 1)
    calibrated_energy = np.polyval(model, data[x_variable].to_numpy())

    # Plot the calibrated spectrum
    plt.figure(figsize=fig_size)
    plt.plot(calibrated_energy, data['counts'], color=color, label='Calibrated Spectrum')

    # Drop vertical lines at identified peaks with automatic colors
    cmap = get_cmap("tab10")  # Use a color map for automatic coloring
    for i, energy in enumerate(energy_values):
        plt.axvline(energy, color=cmap(i % 10), linestyle='--', label=f'{energy} keV')

    # Apply user-specified x-axis limits
    if x_limits:
        plt.xlim(x_limits)

    # Labels and title
    plt.xlabel(x_label, fontsize=font_size)
    plt.ylabel(y_label, fontsize=font_size)
    plt.title(title, fontsize=font_size)
    plt.legend(loc='upper right', fontsize=font_size)
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
    plt.plot(df[x_variable], df[y_variable], color=plot_color, label=legend_label)
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
                                   co60=False, cs137=False, na22=False, ba133=False, lyso=False,
                                   extrapolation_range=(10, 1500), num_points=500, x_label=None, fontsize=16,
                                   y_label=None, save_plot=None, save_file_name=None, plot_dpi=None):
    """
    Performs linear regression on energy-area data, plots the data points, regression line, and confidence interval,
    and highlights peaks for specific isotopes.

    Parameters:
        energy_values (list or array): Known energy values.
        area_values np.ndarray: Corresponding area values.
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
    area = area_values[:, 0]
    area_error = area_values[:, 1]

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
    plt.errorbar(energy, area, area_error, color='blue', capsize=5, alpha=0.25, ls='')
    plt.plot(x_range, area_extrapolated, color=plot_color, linewidth=2, label='Extrapolated Regression Line')
    plt.fill_between(x_range, area_extrapolated - std_dev, area_extrapolated + std_dev, alpha=0.25, color=plot_color)

    # Annotate each data point with its energy value
    for x, y in zip(energy, area):
        plt.text(x, y, f'{x:.2f}', fontsize=10, ha='right')

    if lyso:
        lyso_peaks = [88, 202 + 88, 307 + 88, 202 + 88 + 307]
        for peak_energy in lyso_peaks:
            peak_area = model(peak_energy)
            plt.plot(peak_energy, peak_area, 'o', color='black',
                     label='Co-60 Peaks' if peak_energy == lyso_peaks[0] else "")
            plt.text(peak_energy, peak_area, f'({peak_energy:.1f} keV, {peak_area:.2e} Wb)', color='black', fontsize=10,
                     ha='left')

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


def plot_fwhm_resolution(mu_values, sigma_values, x_label=None, save_plot=None, save_file_name=None,
                         plot_dpi=None, figure_size=None, show_error_bars: bool = True,
                         x_lim_fwhm=None, y_lim_fwhm=None, x_lim_res=None, y_lim_res=None, font_size=16):
    """
    Creates regression plots for FWHM and Resolution against Mu values.

    Parameters
    ----------
    mu_values : array-like
        Array or list containing Mu values in terms of area (e.g., [Wb]).
    sigma_values : array-like
        Array or list containing standard deviation values corresponding to the Mu values.
    x_label: str, optional
        The x-label for the plot
    save_plot: bool, optional
        Whether to save the plot or not.
    save_file_name: str, optional
        The name of the file to save the plot to.
    plot_dpi: int, optional
        The DPI value for the plot.
    figure_size: tuple, optional
        The size of the figure for the plot.
    show_error_bars: bool, optional
        Whether to show the error bars or not.
    x_lim_fwhm: float, optional
        The x-limit for the FWHM plots.
    y_lim_fwhm: float, optional
        The y-limit for the FWHM plots.
    x_lim_res: float, optional
        The x-limit for the resolution plots.
    y_lim_res: float, optional
        The y-limit for the resolution plots.
    font_size: int, optional
        The font size for plot elements.

    Returns
    -------
    None
        Displays the generated regression plots. Saves the figure if a file path is provided.
    """
    # Create a DataFrame for plotting
    mu_errs = mu_values[:, 1]
    sigma_errs = sigma_values[:, 1]
    mu_values = mu_values[:, 0]
    sigma_values = sigma_values[:, 0]

    fwhm, res = calculate_resolution_and_fwhm(sigma_values, mu_values)

    data = pd.DataFrame({'Mu': mu_values,
                         'FWHM': fwhm,
                         'Resolution': res})

    # Create the regression plots
    f, ax = plt.subplots(1, 2, figsize=figure_size)

    # First plot: Mu vs FWHM
    sns.regplot(x='Mu', y='FWHM', data=data, ci=None, scatter_kws={'s': 100}, line_kws={'color': 'red'}, ax=ax[0],
                color='#1f77b4')
    if show_error_bars:
        ax[0].errorbar(data['Mu'], data['FWHM'],
                       xerr=mu_errs, yerr=np.std(data['FWHM'].to_numpy()),
                       ls='', capsize=5, color='#1f77b4')
    ax[0].set_xlabel(x_label, fontsize=font_size)
    ax[0].set_ylabel('FWHM', fontsize=font_size)
    ax[0].set_title('Full Width Half Maximum', fontsize=font_size)
    ax[0].grid(False)
    ax[0].set_xlim(x_lim_fwhm)
    ax[0].set_ylim(y_lim_fwhm)

    # Second plot: Mu vs Resolution
    sns.regplot(x='Mu', y='Resolution', data=data, ci=None, scatter_kws={'s': 100}, line_kws={'color': 'green'},
                ax=ax[1], color='#1f77b4')
    if show_error_bars:
        ax[1].errorbar(data['Mu'], data['Resolution'],
                       xerr=mu_errs, yerr=np.std(data['Resolution'].to_numpy()),
                       ls='', capsize=5, color='#1f77b4')
    ax[1].set_xlabel(x_label, fontsize=font_size)
    ax[1].set_ylabel('Resolution', fontsize=font_size)
    ax[1].set_title('Energy Resolution %', fontsize=font_size)
    ax[1].grid(False)
    ax[1].set_xlim(x_lim_res)
    ax[1].set_ylim(y_lim_res)
    f.tight_layout()

    # Save the figure if a save path is provided
    if save_plot:
        plt.savefig(save_file_name, dpi=plot_dpi)
    plt.show()


####################################################################################################################

def process_and_plot_spectra(signal, background, signal_duration, background_duration, xlabel=None, ylabel=None,
                             title=None, figure_size=None, scale=False, xlims=None, logscale_x=False, logscale_y=False):
    """
    Process and plot spectra for signal and background data. Interpolates data,
    scales the background to match the signal, subtracts the background, and plots
    the signal, background, and corrected spectra.

    Parameters:
    -----------
    signal : pandas.DataFrame
        DataFrame containing 'area' and 'counts' columns for the signal spectrum.
    background : pandas.DataFrame
        DataFrame containing 'area' and 'counts' columns for the background spectrum.
    signal_duration : int
        Duration of signal data collection in seconds.
    background_duration : int
        Duration of background data collection in seconds.

    Returns:
    --------
    common_x : numpy.ndarray
        Common x-axis (area) used for interpolation.
    corrected_signal : numpy.ndarray
        The background-subtracted and normalized signal spectrum.
    """
    # Determine the overlapping area range
    overlap_min = min(signal['area'].min(), background['area'].min())
    overlap_max = max(signal['area'].max(), background['area'].max())

    # Define a common x-axis within the overlapping range
    common_x = np.linspace(overlap_min, overlap_max, 10_000)

    # Interpolate the signal and background spectra
    # we create interpolation function here
    signal_interp = interp1d(signal['area'], signal['counts'], kind='nearest', fill_value="extrapolate")
    background_interp = interp1d(background['area'], background['counts'], kind='nearest', fill_value="extrapolate")

    # Evaluate the interpolated values on the common x-axis
    # we generate the interpolated y values from the newly generated x values using interpolation function
    signal_values = signal_interp(common_x)
    background_values = background_interp(common_x)

    # Scale the background to match the signal's peak
    scaling_factor = signal_values.max() / background_values.max() if scale else 1
    background_scaled = background_values * scaling_factor

    # Calculate the normalized and background-subtracted signal
    # we divide the signal with duration to get counts/s
    corrected_signal = (signal_values / signal_duration) - (background_scaled / background_duration)

    # we do not want negative area values, so we keep only x > 0
    mask_ = common_x > 0
    common_x = np.where(mask_, common_x, np.nan)
    # apply the mask on the good values x > 0
    sig = np.where(mask_, signal_values / signal_duration, np.nan)
    bkg = np.where(mask_, background_values / background_duration, np.nan)
    corr = np.where(mask_, corrected_signal, np.nan)

    # scale the axis to log
    if logscale_x:
        common_x = np.log10(common_x)
    if logscale_y:
        sig = np.log10(sig)
        bkg = np.log10(bkg)
        corr = np.log10(corr)

    # Plot the spectra
    plt.figure(figsize=figure_size)

    plt.plot(common_x, sig, label='Signal', color='blue')
    plt.plot(common_x, bkg, label='Background', color='red')
    plt.plot(common_x, corr, label='Corrected Signal', color='green')

    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.xlim(xlims)
    plt.show()

    # it is giving the x values and the background subtracted signal
    return common_x, corrected_signal
