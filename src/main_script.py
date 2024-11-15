from EnergyCalibrationPy import load_data, plot_calibrated_spectrum

# Path to your CSV file
file_path = './SubtractedAreaHist_30V_OnsemiSiPM+CsI+Co60+RoomTemp.csv'

# Load the data
data = load_data(file_path, column_names=['area', 'counts'])  # Adjust column names if necessary

# Known calibration data
energy_values = [88, 290, 395, 597]  # Known energy values in keV
area_values = [3.51e-08, 6.88e-08, 8.13e-08, 1.27e-07]  # Corresponding detector responses (area or amplitude)

# Plot customization options
x_limits = (0, 2000)  # Set x-axis limits
color = 'red'  # Color of the spectrum plot
title = 'Energy Calibration - Customized Spectrum'  # Title of the plot
xlabel = 'Energy [keV]'  # X-axis label
ylabel = 'Counts'  # Y-axis label
figsize = (12, 8)  # Default figure size; change if desired

# Plot the calibrated spectrum
plot_calibrated_spectrum(data=data,
                         energy_values=energy_values,
                         area_values=area_values,
                         x_limits=x_limits,
                         color=color,
                         title=title,
                         xlabel=xlabel,
                         ylabel=ylabel,
                         figsize=figsize)
