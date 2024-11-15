import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

def plot_calibrated_spectrum(
        data, 
        energy_values, 
        area_values, 
        x_limits=None, 
        color='blue', 
        title='Energy Calibration',
        xlabel='Energy [keV]', 
        ylabel='Counts', 
        figsize=(12, 8)
    ):
    """
    Plots the calibrated energy spectrum with user-defined settings.
    
    Parameters:
        data (pd.DataFrame): Data containing 'x' and 'counts' columns.
        energy_values (list or array): Known energy values for calibration.
        area_values (list or array): Corresponding area values for calibration.
        x_limits (tuple or None): Tuple defining x-axis limits (xmin, xmax). Default is None.
        color (str): Color of the calibrated spectrum plot.
        title (str): Title of the plot.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        figsize (tuple): Size of the figure. Default is (12, 8).
    """
    if len(energy_values) != len(area_values):
        raise ValueError("Energy values and area values must have the same length.")
    
    # Perform linear calibration (regression)
    model = np.polyfit(area_values, energy_values, 1)
    calibrated_energy = np.polyval(model, data['area'].to_numpy())
    
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
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout()
    plt.show()

