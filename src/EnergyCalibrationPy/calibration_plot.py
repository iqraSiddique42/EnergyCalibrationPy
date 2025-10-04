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
                             y_label='Counts/s',
                             data_label='Calibrated spectrum',
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
    # values = values[:, 0]

    if len(energy_values) != len(values):
        raise ValueError("Energy values and area values must have the same length.")

    # Perform linear calibration (regression)
    model = np.polyfit(values, energy_values, 1)
    calibrated_energy = np.polyval(model, data[x_variable].to_numpy())

    # Plot the calibrated spectrum
    f, ax = plt.subplots(figsize=fig_size)
    # ax.figure(figsize=fig_size)
    ax.plot(calibrated_energy, data['counts'], color=color, label=data_label)

    # Drop vertical lines at identified peaks with automatic colors
    # cmap = get_cmap("tab10")  # Use a color map for automatic coloring
    # for i, energy in enumerate(energy_values):
    #     ax.axvline(energy, color=cmap(i % 10), linestyle='--', label=f'{energy} keV')

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
    # plt.show()

    return ax, model


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

def regression_and_plot_with_peaks(
    energy_values,
    area_values,
    sigma_values,
    title='Regression Plot Energy → Area',
    plot_color='red',
    co60=False, cs137=False, na22=False, ba133=False, lyso=False,
    extrapolation_range=(10, 1500), num_points=500,
    x_label=None, y_label=None,
    fontsize=16,
    tick_size=14, tick_length_major=7, tick_length_minor=4, tick_width=1.1,
    enable_minor_ticks=True,
    # --- NEW: CI (confidence band) controls ---
    show_ci=True, ci_alpha=0.25, ci_in_legend=False, ci_label='±1σ (residual)',
    save_plot=None, save_file_name=None, plot_dpi=None, save_data=False,
    color_marker=None, elements=None
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    energy = np.array(energy_values)
    area = np.array(area_values)
    area_error = np.array(sigma_values)

    # Linear regression
    coefficients = np.polyfit(energy, area, 1)
    model = np.poly1d(coefficients)

    # Extrapolated line
    x_range = np.linspace(extrapolation_range[0], extrapolation_range[1], num_points)
    area_extrapolated = model(x_range)

    # Residual std for ±1σ band
    y_pred = model(energy)
    residuals = area - y_pred
    std_dev = np.std(residuals, ddof=1) if len(residuals) > 1 else 0.0

    # Plot
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # Scatter with labels
    c_ = color_marker if color_marker is not None else ['C0'] * len(energy)
    e_ = elements if elements is not None else [''] * len(energy)
    for E, A, col, lab in zip(energy, area, c_, e_):
        ax.scatter(E, A, color=col, label=(f'{lab}: {E} keV' if lab else None), zorder=10)

    # Error bars
    ax.errorbar(x=energy, y=area, yerr=area_error, color='blue', capsize=5, ls='', zorder=9)

    # Regression line
    ax.plot(x_range, area_extrapolated, color=plot_color, linewidth=2, zorder=8)

    # --- CI band (now controllable; NOT in legend by default) ---
    if show_ci and std_dev > 0:
        band_kwargs = dict(alpha=ci_alpha, color=plot_color, zorder=7)
        if ci_in_legend:
            band_kwargs['label'] = ci_label
        else:
            band_kwargs['label'] = '_nolegend_'   # Matplotlib ignores this in legend
        ax.fill_between(x_range, area_extrapolated - std_dev,
                        area_extrapolated + std_dev, **band_kwargs)

    # Grid + ticks
    ax.grid(True, alpha=0.25, ls='--')
    if enable_minor_ticks:
        ax.minorticks_on()
    ax.tick_params(axis='both', which='major', labelsize=tick_size,
                   length=tick_length_major, width=tick_width)
    ax.tick_params(axis='both', which='minor', labelsize=max(tick_size-2, 1),
                   length=tick_length_minor, width=max(tick_width-0.2, 0.5))

    # --- Isotope annotations ---
    if lyso:
        lyso_peaks = [88, 290, 395, 597]
        first = True
        for pk in lyso_peaks:
            ypk = model(pk)
            ax.plot(pk, ypk, 'o', color='black', label='LYSO (proj.)' if first else None, zorder=11)
            ax.text(pk, ypk, f'({pk:.1f} keV, {ypk:.2e} Wb)', color='black',
                    fontsize=max(fontsize-4, 8), ha='left', va='bottom')
            first = False
    if co60:
        first = True
        for pk in [1173.2, 1332.5]:
            ypk = model(pk)
            ax.plot(pk, ypk, 'o', color='black', label='Co-60 Peaks' if first else None, zorder=11)
            ax.text(pk, ypk, f'({pk:.1f} keV, {ypk:.2e} Wb)', color='black',
                    fontsize=max(fontsize-4, 8), ha='left', va='bottom')
            first = False
    if cs137:
        pk = 662
        ypk = model(pk)
        ax.plot(pk, ypk, 'o', color='maroon', label='Cs-137 Peak', zorder=11)
        ax.text(pk, ypk, f'({pk} keV, {ypk:.2e} Wb)', color='maroon',
                fontsize=max(fontsize-4, 8), ha='left', va='bottom')
    if na22:
        first = True
        for pk in [511, 1274]:
            ypk = model(pk)
            ax.plot(pk, ypk, 'o', color='cyan', label='Na-22 Peaks' if first else None, zorder=11)
            ax.text(pk, ypk, f'({pk:.1f} keV, {ypk:.2e} Wb)', color='cyan',
                    fontsize=max(fontsize-4, 8), ha='left', va='bottom')
            first = False
    if ba133:
        first = True
        for pk in [81, 356]:
            ypk = model(pk)
            ax.plot(pk, ypk, 'o', color='magenta', label='Ba-133 Peaks' if first else None, zorder=11)
            ax.text(pk, ypk, f'({pk:.1f} keV, {ypk:.2e} Wb)', color='magenta',
                    fontsize=max(fontsize-4, 8), ha='left', va='bottom')
            first = False

    # Labels & (optional) title
    ax.set_xlabel(x_label if x_label else 'Energy [keV]', fontsize=fontsize)
    ax.set_ylabel(y_label if y_label else 'Area [Wb]', fontsize=fontsize)
    if title:
        ax.set_title(title, fontsize=fontsize)

    # De-duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    dedup = {}
    for h, l in zip(handles, labels):
        if l and l not in dedup and l != '_nolegend_':
            dedup[l] = h
    if dedup:
        ax.legend(dedup.values(), dedup.keys(), loc='best', fontsize=max(fontsize-2, 8))

    plt.tight_layout()

    # Save data
    if save_data:
        ene_area = pd.DataFrame({'energy': energy, 'area': area, 'area_error': area_error})
        extrapolated = pd.DataFrame({'x_range': x_range,
                                     'area_extrapolated': area_extrapolated,
                                     'error': np.full_like(x_range, std_dev)})
        ene_area.to_csv('./energy_area.csv', index=False)
        extrapolated.to_csv('./extrapolated_area.csv', index=False)

    if save_plot:
        plt.savefig(save_file_name, dpi=plot_dpi)

    plt.show()
    return model

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
    mu_values = mu_values
    sigma_values = sigma_values

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
                       xerr=sigma_values, yerr=np.std(data['FWHM'].to_numpy()),
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
                       xerr=sigma_values, yerr=np.std(data['Resolution'].to_numpy()),
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
    mask_ = common_x  # > 0
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
