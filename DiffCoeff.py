#!/usr/bin/env python
# coding: utf-8

#Calculate the diffusion coefficient from velocity autocorrelation function using Green-Kubo relation

# Import necessary libraries
import os
import numpy as np
import pandas as pd


# Define the subdirectory path where your data is located
subdirectory = './'

# Get a list of files starting with "VACF" in the subdirectory
file_list = [filename for filename in os.listdir(subdirectory) if filename.startswith('VACF')]

# Initialize empty lists to store results and error files
results = []
error_files = []

# Iterate over each file in the list
for filename in file_list:
    # Construct the full file path
    filepath = os.path.join(subdirectory, filename)
    
    # Load the data from the text file
    vacf_data = np.loadtxt(filepath)
    
    # Extract time lags and velocity autocorrelation function from the data
    time_lags = vacf_data[:, 0] * 1e-15
    vel_auto_corr = vacf_data[:, 1]
    
    # Calculate the diffusion coefficient using the Green-Kubo relation
    dt = time_lags[1] - time_lags[0]
    diff_coeff = np.trapz(vel_auto_corr, dx=dt)
    diff_coeff *= 1 / 3
    diff_coeff *= 1e+4
    
    # Store the result in a dictionary
    filename = filename.replace('VACF_CUSTOMIZED_', '')
    result = {'Crystal': filename, 'Diffusion Coefficient (cm^2/s)': diff_coeff}
    results.append(result)
    
    # Check if diff_coeff is null (NaN)
    if np.isnan(diff_coeff):
        error_files.append(filename)

# Create a DataFrame from the results
df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
df.to_csv('diffusion_coefficients.csv', index=False)




