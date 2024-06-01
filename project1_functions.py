# -*- coding: utf-8 -*-
"""
@author Tomasz Lasota 01 Jan 2024

Automated processing of txt files containing temperature measurements over time (see README.md). 
This script must be run in conjunction with project1_main.py. 

version 1.0

"""




import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import re
import os


# function for extracting indexes and data for each temperature hold 
def find_last_plateau_indexes(df, plateau_length, plateau_threshold, num_datapoints, target_temperatures=None, target_channels=None):
    plateau_indexes = {(temp, ch): [] for temp in target_temperatures for ch in target_channels}
    print("############Commencing search for plateau indices################")
    for temp in target_temperatures:
        for ch in target_channels:
            # Calculate rolling mean with a window size equal to plateau_length
            rolling_mean = df[ch].rolling(window=plateau_length, min_periods=1).mean()
            # Find indices where the difference between rolling mean and current value is within plateau_threshold
            plateau_indices = df[(abs(rolling_mean - df[ch]) <= plateau_threshold) & (abs(df[ch] - temp) <= plateau_threshold)].index
            # Take the last num_datapoints indices (skip last 10 indexes to avoid the post hold cooling phase)
            last_plateau_indices = plateau_indices[-num_datapoints:-10]
            if len(last_plateau_indices) == num_datapoints - 10:
                # Add the last plateau indices to the dictionary
                plateau_indexes[(temp, ch)] = last_plateau_indices.tolist()
            else:
                plateau_indexes[(temp, ch)] = []
                print(f"failed to find plauteaus for {temp} degC in {ch}")
    return plateau_indexes

# function to extract time and temperature data using pk and trough indices (this extracts only either cool or heat depending on start end orientation)
# 
def extract_data(start_index, end_index, dataset, channels):
    extracted_data = {}
    for channel in channels:
        for i, (start, end) in enumerate(zip(start_index[channel], end_index[channel])):
            cycle_time = dataset["time"].iloc[start:end+1]
            cycle_time = cycle_time.reset_index(drop=True)
            cycle_time_normalised = normalize_list(cycle_time)                                         # might not be needed anymore
            cycle_temperature = dataset[channel].iloc[start:end+1]
            cycle_temperature = cycle_temperature.reset_index(drop=True)
            extracted_data[(channel, f"cycle_{i +1}")] = {"Time: ": cycle_time_normalised,
                                                "Temperature: ": cycle_temperature}
    return extracted_data

# concatonate data from all cycles for each channel
def concat_cycles(dict):
    concat_data = {}
    for (channel, cycle), values in dict.items():
        if channel not in concat_data:
            concat_data[channel] = {"Time: ": [], "Temperature: ": []}                                  # the passed in dictionary contains a list as a key
        concat_data[channel]["Time: "].extend(values["Time: "])
        concat_data[channel]["Temperature: "].extend(values["Temperature: "])
    return concat_data

# function normalising time series to the first value
def normalize_list(list):
    new_list = []
    for i in range(len(list)):
        value = list[i] - list[0]
        value = "{:.2f}".format(value)
        new_list.append(value)
    return new_list

# function for reversing indexes stored in a dict of channel index values
def reverse_index(dict):
    new_dict = {}
    for key, value in dict.items():
        new_dict[key] = value[::-1]
    return new_dict

# function for removing data at a specific index position from a dictionary
def remove_data(dict, index):
    for key, value in dict.items():
        dict[key] = np.delete(value, index)
    return dict

# function for finding sync t_start time
def get_t_start(df, channel, T_sync_start, T_sync_end, T_start):
    t_start = []
    i_sync_start = next(x for x, val in enumerate(df[channel])
                                if val > T_sync_start)
    i_sync_end = next(x for x, val in enumerate(df[channel])
                                if val > T_sync_end)
    times = df["time"][i_sync_start:i_sync_end]                                                     # times with values ranging from 35 to 45
    temps = df[channel][i_sync_start:i_sync_end]                                                    # corresponding temperature values for channel 1
    
    t_start.append(np.interp(T_start,temps,times))                                                  # this finds the time at which 40 deg C is reached (interpolation needed as the exact 40 degC might
    return t_start                                                                                  # not exist

# function for finding indexes of peaks returning a dict of indices per channel
def find_peaks_indices(df, channels, height, width, plateau_size):
    peaks_indices = {}
    for channel in channels:
        temp_pk_indices = find_peaks(df[str("ch"+str(channel+1))],                                  # find peaks matching certain criteria 
                                      height = height, width = width, plateau_size=plateau_size)[0]
        if(len(temp_pk_indices) > len(peaks_indices)):
            peaks_indices[f"ch{channel+1}"] = temp_pk_indices
    return peaks_indices

# function for finding indexes of troughs
def find_troughs_indices(df, channels, width, plateau_size, height):
    trough_indices = {}
    for channel in channels:
        temp_trough_indices = find_peaks(-df[str("ch"+str(channel+1))],                              # find peaks matching certain criteria 
                                      width = width, plateau_size=plateau_size, height=height)[0]
        if(len(temp_trough_indices) > len(trough_indices)):
            trough_indices[f"ch{channel+1}"] = temp_trough_indices
    return trough_indices
# function for extracting max peak temp and time values based on peak indices returing a dict of values
# for each channel
def get_peak_XY(df, pk_indices):
    peaks_x1 = {}
    peaks_y1 = {}
    for ch_pk in pk_indices:
        for val in pk_indices.values():
            peaks_y1[ch_pk] = df[ch_pk][val]                                                              # appends the corresponding data for each index found by find_peaks()
            peaks_x1[ch_pk] = df['time'][val]                                                               # this uses timestamp already converted to seconds
    return peaks_x1, peaks_y1
# function for extracting max trough temp and time values based on trough indices returning a dict of values
def get_trough_XY(df, trough_indices):
    troughs_x1 = {}
    troughs_y1 = {}
    for ch_troughs in trough_indices:
        for val in trough_indices.values():
            troughs_y1[ch_troughs] = df[ch_troughs][val]
            troughs_x1[ch_troughs] = df['time'][val]   
    return troughs_x1, troughs_y1

# function extracting ALL SERIES values of time and temperature based on channel and temp from dictionary
def extract_peak_XY(dict, df, channel, temperature):
    peaks_x1 = []
    peaks_y1 = []
    for i in dict[(temperature, channel)]:
        peaks_y1.append(df[channel][i])                                                               # appends the corresponding data for each index found by find_peaks()
        peaks_x1.append(df['time'][i])                                                                # this uses timestamp already converted to seconds
    return peaks_x1, peaks_y1

def biexponential(t, T0, T1, t1, T2, t2):
    t = np.asarray(t, dtype=float)
    return T0 + T1 * np.exp(-t/t1) + T2 * np.exp(-t/t2)

# function for fitting biexponential model
def fit_biexponential(dict, initial_guess):
    popt_dict = {}
    for key, values in dict.items():
        # get timestamp and temperature values from the dictionary
        times = np.array(values["Time: "])
        temperatures = np.array(values["Temperature: "])
        popt_biexp, _ = curve_fit(biexponential, times, temperatures, p0=initial_guess, maxfev=10000)
        popt_dict[key] = popt_biexp

    return popt_dict

# function for testing the biexponential fit
def test_biexponential_fit(x, y, parameters, plot_models, channel):
    T0, T1, t1, T2, t2 = parameters
    y_fit = biexponential(x, T0, T1, t1, T2, t2)
    if plot_models:
        plt.scatter(x, y, label= "Data")
        plt.plot(x, y_fit, label="Fit", color="red")
        plt.xlabel("Time (s)")
        plt.ylabel("Temperature (°C)")
        plt.title(f"Biexponential fit for {channel}")
        plt.legend()
        plt.show()
    # Calculate R-squared
    residuals = y - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f'R-squared for {channel}: {r_squared}')

    # Calculate RMSE
    rmse = np.sqrt(np.mean(residuals**2))
    print(f'RMSE for {channel}: {rmse}')
    
    return r_squared, rmse

# function to find time of a specifict temperature based on the fitted biexponential
def time_to_temp(target_temp, parameters):
    T0, T1, t1, T2, t2 = parameters
    def temperature_difference(x):
        return biexponential(x, T0, T1, t1, T2, t2) - target_temp
    
    t = root_scalar(temperature_difference, bracket=[-100,110])
    time = t.root

    return time

# function for finding min, max, mean values returining a dictionary for all temperatures and channels
def find_min_max_avg(dict, channels, temperatures):
    min_dict = {}
    max_dict = {}
    avg_dict = {}
    for temp in temperatures:
        for ch in channels:
            if len(dict[(ch, temp)]["Temperature: "]) > 0:
                min_dict[(ch, temp)] = np.min(dict[(ch, temp)]["Temperature: "])
                max_dict[(ch, temp)] = np.max(dict[(ch, temp)]["Temperature: "])
                avg_dict[(ch, temp)] = np.mean(dict[(ch, temp)]["Temperature: "])
    return min_dict, max_dict, avg_dict

def get_min(dict):
    min_dict = {}
    unique_keys = set(key[1] for key in dict.keys())

    for key in unique_keys:
        values_for_key = [value for k, value in dict.items() if k[1] == key]
        min_dict[key] = np.min(values_for_key)
    return min_dict

def get_max(dict):
    max_dict = {}
    unique_keys = set(key[1] for key in dict.keys())

    for key in unique_keys:
        values_for_key = [value for k, value in dict.items() if k[1] == key]
        max_dict[key] = np.max(values_for_key)
    return max_dict

def get_avg(dict):
    avg_dict = {}
    unique_keys = set(key[1] for key in dict.keys())

    for key in unique_keys:
        values_for_key = [value for k, value in dict.items() if k[1] == key]
        avg_dict[key] = np.mean(values_for_key)
    return avg_dict


# function for combining results from time_to_temp
def combine_results(dict, channels, start, end):
    results_dict = {}
    for channel in channels:
        temp_start = time_to_temp(start, dict[channel])
        temp_end = time_to_temp(end, dict[channel])
        results_dict[channel] = {"Time_start": temp_start, "Time_end": temp_end}
    return results_dict

# function for sorting lists
def sort_dict(dict, channels):
    sorted_dict = {}
    for ch in channels:
        time = dict[ch]["Time: "]
        temperature = dict[ch]["Temperature: "]
        time_temp_pair = list(zip(time, temperature))
        time_temp_pair.sort(key=lambda x: x[0])
        # separate into lists after sorting
        time, temperature = zip(*time_temp_pair)
        sorted_time = list(time)
        sorted_temperature = list(temperature)
        sorted_dict[ch] = {"Time: ": sorted_time, "Temperature: ": sorted_temperature}
    
    return sorted_dict

# function for calculateing heat and cool rates
def calculate_rates(dict, channels, start_temp, end_temp):
    results = {}
    for ch in channels:
        start_time = dict[ch]["Time_start"]
        end_time = dict[ch]["Time_end"]
        rate = (end_temp - start_temp)/(end_time - start_time)
        results[ch] = {"Rate": rate}
    
    return results

# function for loading text files from the current directory
def get_txt_files():
    dir = os.path.dirname(os.path.realpath(__file__))
    list_file_names = [file for file in os.listdir(dir) if file.endswith(".txt")]

    return list_file_names

# function for extracting instrument ID from the file name
def get_instrument_ID(file_name):
    pattern = r"\d{11}"
    match = re.search(pattern, file_name)
    instrument_ID = ""
    if match:
        instrument_ID = match.group()
    else:
        print("Instrument ID not found in the file name: " + file_name)
    
    return instrument_ID

# function for extracting date from loaded txt files 
def get_date(file_name):
    pattern = r"\d{4}-\d{2}-\d{2}"
    test_date = ""
    match = re.search(pattern, file_name)
    if match:
        test_date = match.group()
    else:
        print("Date not found in the file name: " + file_name)
    
    return test_date


if __name__ == "__main__":
    print("This is a supplementary module containing function definitions only. It must be run using project1_main.py")