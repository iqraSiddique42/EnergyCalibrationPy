from matplotlib import pyplot as plt
from pymultifit.fitters import GaussianFitter


def generate_colors(num_colors, cmap_name="tab10"):
    """
    Generate a list of colors from a specified color map.
    
    Parameters:
        num_colors (int): Number of colors to generate.
        cmap_name (str): Name of the color map to use (default is "tab10").
        
    Returns:
        list: List of color values.
    """
    from matplotlib.cm import get_cmap
    cmap = get_cmap(cmap_name)
    return [cmap(i % cmap.N) for i in range(num_colors)]


def validate_calibration_data(energy_values, area_values):
    """
    Validates that the calibration data lists have the same length.
    
    Parameters:
        energy_values (list or array): Known energy values.
        area_values (list or array): Corresponding area values.
    
    Raises:
        ValueError: If the lengths of energy_values and area_values do not match.
    """
    if len(energy_values) != len(area_values):
        raise ValueError("Energy values and area values must have the same length.")


import os


def check_file_exists(file_path):
    """
    Checks if a file exists at the given path.
    
    Parameters:
        file_path (str): Path to the file.
    
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")


def convert_weber_to_volts(weber_value, conversion_factor=1e6):
    """
    Converts a value from Weber to Volts using a specified conversion factor.
    
    Parameters:
        weber_value (float): The value in Webers.
        conversion_factor (float): Factor to convert Weber to Volts (default is 1e6).
        
    Returns:
        float: The converted value in Volts.
    """
    return weber_value * conversion_factor


def format_axis(ax, xlabel, ylabel, fontsize=14):
    """
    Applies consistent formatting to a matplotlib axis.
    
    Parameters:
        ax (matplotlib.axes.Axes): Axis object to format.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        fontsize (int): Font size for labels.
    """
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize - 2)


###########################################################################

def calculate_resolution_and_fwhm(sigma, mu):
    """
    Calculate the energy resolution as a percentage.

    Parameters:
    sigma (float): Standard deviation of the Gaussian peak.
    mu (float): Centroid (mean) of the peak.

    Returns:
    float: Resolution as a percentage.
    """
    # Calculate FWHM from sigma
    fwhm = 2.355 * sigma

    # Calculate resolution
    resolution = (fwhm / mu) * 100
    return fwhm, resolution


###########################################################################

def gaussian_fitter(dataframe, x, y, p0, normalization=1.) -> GaussianFitter:
    temp_ = dataframe.copy(deep=True)
    temp_[x] = temp_[x] / normalization

    fitter = GaussianFitter(len(p0), temp_[x], temp_[y])
    fitter.fit(p0)
    fitter.plot_fit(show_individual=True, auto_label=True)
    plt.show()

    return fitter


def parameter_extractor(fitter, gaussian_numbers, normalization=1.):
    amp_, mu_, std_ = fitter.get_parameters(True, True, True, gaussian_numbers)

    mu_ = mu_ * normalization
    std_ = std_ * normalization

    return amp_, mu_, std_
