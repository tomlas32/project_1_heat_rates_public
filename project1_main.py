# -*- coding: utf-8 -*-
"""
@author Tomasz Lasota 01 Jan 2024

Automated processing of txt files containing temperature measurements over time (see README.md). 
This script must be run in conjunction with project1_functions.py. 

version 1.0

"""

import pandas as pd
import matplotlib.pyplot as plt
from project1_functions import *

### USER to modify as per individual needs ###
plot_TSC = False
plot_peak_finder = True
test_models = True
plot_models = False
print_holds = False
channel_plot = "ch1"                # choose channel to plot
hold_pass_range = 1.5               # pass/fail range threshold for temperature holds (threshold variation between read and set temperature values)


### Do NOT modify any of the following code below ###
# initilise variables
PCR_channels = [0, 1, 2, 3]
channels = ["ch1", "ch2", "ch3", "ch4"]
target_temperatures = [50, 60, 75, 95]
T_start = 40                        # threshold temperature for start time to synchronsise (at the start of RT)
T_sync_start = 35
T_sync_end = 45
start_temp = 60                     # temperature to start caluclating time for heating/cooling rate
end_temp = 95                       # target temperature for heating/cooling rate
p0_heat = (400,-50,0.3,-300,-3e7)   # Initial guess for heating biexp fit coeffs
p0_cool = (50,40,0.5,-5,-2)         # Initial guess for cooling biexp fit coeffs
results = pd.DataFrame(columns=["Date", "Instrument_id", "Channel1_HR", "Channel2_HR", "Channel3_HR", "Channel4_HR", "Channel1_CR", "Channel2_CR", "Channel3_CR",
                                 "Channel4_CR", "Heat_R-sq", "Heat_RMSE", "Cooling_R-sq", "Cooling_RMSE", "95min", "95max", "95avg", "75min", "75max", "75avg", "60min", "60max", "60avg",
                                   "50min", "50max", "50avg"])

files = get_txt_files()             # function for loading all txt files present in the working dir

for i, file in enumerate(files):

    if file:

        print(f"Processing file {file}")     

        instrument_id = get_instrument_ID(file)
        test_date = get_date(file)

        ### Load data ###
        data=pd.read_csv(file, skiprows=1, skipfooter=1, 
                                    names=['ch1', 'ch2', 'ch3', 'ch4', 'time'], 
                                    header=None, engine='python')
        try:
            ### Sync data to 40 deg C ###
            t_start = get_t_start(data, channel="ch1", T_sync_start=T_sync_start, T_sync_end=T_sync_end, T_start=T_start)   # time at which 40 would be reached                                                                                       
            data["time"] = (data["time"]-t_start)/1000                                                                      # time converted to seconds after the 40deg mark / all values are reduced by t_start
            
            ### Find peaks and troughs indices ###
            pk_indices = find_peaks_indices(data, PCR_channels, 90, width=(40, 180), plateau_size=0)
            trough_indices = find_troughs_indices(data, PCR_channels, height= -90, width=(80, 150), plateau_size=0)

            ### Extract time (x) and temperature (y) values for peaks and trough for a all channels ###
            peaks_x1, peaks_y1 = get_peak_XY(data, pk_indices)
            troughs_x1, troughs_y1 = get_trough_XY(data, trough_indices)

            ### Get heat and cooling rates ###
            data_cooling = extract_data(pk_indices, trough_indices, dataset=data, channels=channels)                            # get data for calculating cooling rates for all channels
            # remove first index value from pk_inices and last from trough_indices for getting heating data (one less cycle for heating than in cooling)
            heat_trough = remove_data(trough_indices, -1)
            heat_pk = remove_data(pk_indices, 0)
            # get heating data
            # NOTE: heating data is has 1 less cycle than in cooling due to the plateau phases on both sides of the cycling
            data_heating = extract_data(heat_trough, heat_pk, dataset=data, channels=channels)
            # concat all the data to fit the biexponential model to (for both cooling and heating)
            heating_data = concat_cycles(data_heating)
            cooling_data = concat_cycles(data_cooling)
            # sort the data
            sorted_heating_data = sort_dict(heating_data, channels)
            sorted_cooling_data = sort_dict(cooling_data, channels)

            # fit biexponential model for heating data for each cycle and channel
            heat_param = fit_biexponential(sorted_heating_data, p0_heat)
            cool_param = fit_biexponential(sorted_cooling_data, p0_cool)
        except ValueError:
            print(f"Error while processing file {file}")
            break
    
        # validate the fit
        if test_models:
            cool_RMSE = {}
            cool_Rsq = {}
            heat_RMSE = {}
            heat_Rsq = {}
            for channel in channels:
                param_heat = cool_param[channel]
                param_cool = heat_param[channel]
                # returns R-squared and RMSE for both models and all channels
                cool_Rsq[channel], cool_RMSE[channel] = test_biexponential_fit(sorted_cooling_data[channel]["Time: "], sorted_cooling_data[channel]["Temperature: "], param_heat, plot_models, channel)
                heat_Rsq[channel], heat_RMSE[channel] = test_biexponential_fit(sorted_heating_data[channel]["Time: "], sorted_heating_data[channel]["Temperature: "], param_cool, plot_models, channel)
        else:
            heat_Rsq, heat_RMSE = "Model not tested", "Model not tested"
            cool_Rsq, cool_RMSE = "Model not tested", "Model not tested"
        
        # fint time to reach target temperature for heating and cooling for each channel
        time_heat = combine_results(heat_param, channels, start_temp, end_temp)
        try:
            time_cool = combine_results(cool_param, channels, end_temp, start_temp)
        except ValueError as e:
            if "f(a) and f(b) must have different signs" in str(e):
                print(f"Warning: No root found for cooling dataset in file {file}")
                time_cool = None
            else:
                raise

        # calculate heat and cool rates
        heat_rates = calculate_rates(time_heat, channels, start_temp, end_temp)
        if time_cool is not None:
            cool_rates = calculate_rates(time_cool, channels, end_temp, start_temp)
        else:
            cool_rates = {}
            for ch in channels:
                cool_rates[ch] = {"Rate": "Failed to determine"}

        ### Extract data for temperature holds "ALL CHANNELS"###
        holds_data = find_last_plateau_indexes(data, plateau_length=500, plateau_threshold=hold_pass_range, num_datapoints=500, target_temperatures=target_temperatures, target_channels=channels)
        result = {}
        for ch in channels:
            for temp in target_temperatures:
                peaks_x, peaks_y = extract_peak_XY(holds_data, data, ch, temp)
                result[(ch, temp)] = {"Time: ": peaks_x, "Temperature: ": peaks_y}
        ### Calculate min, max, mean values for all temp holds ###
        # calculate min and max values for each temperature hold and channel that met the plateau threshold criteria and returns a dict of values
        min_dict, max_dict, avg_dict = find_min_max_avg(result, channels, target_temperatures) 
        min_dict_val = get_min(min_dict)
        max_dict_val = get_max(max_dict)
        avg_dict_val = get_avg(avg_dict)

        if print_holds:
            for ch in channels:
                for temp in target_temperatures:
                    if len(result[(ch, temp)]["Time: "]) > 0 :
                        print(f"{ch} {temp} min: {min_dict[(ch, temp)]} max: {max_dict[(ch, temp)]} avg: {avg_dict[(ch, temp)]}")
                    else:
                        print(f"Failed to find min, max, avg for channel: {ch} at {temp} degC hold")


        # fill dataframe with data
        results.loc[i] = [test_date, instrument_id, heat_rates["ch1"]["Rate"], heat_rates["ch2"]["Rate"], heat_rates["ch3"]["Rate"], heat_rates["ch4"]["Rate"],
                   cool_rates["ch1"]["Rate"], cool_rates["ch2"]["Rate"], cool_rates["ch3"]["Rate"], cool_rates["ch4"]["Rate"], heat_Rsq, heat_RMSE, cool_Rsq, cool_RMSE,
                   min_dict_val[95], max_dict_val[95], avg_dict_val[95], min_dict_val[75], max_dict_val[75], avg_dict_val[75], min_dict_val[60], max_dict_val[60], avg_dict_val[60],
                   min_dict_val[50], max_dict_val[50], avg_dict_val[50]]

        ############### Plotting TSC data and peaks and troughs ###############
        if plot_TSC: 
            plt.figure(figsize=(12,8))
            plt.title(file)
            plt.plot(data["time"], data[channel_plot], color="black")                                             # need to add other curves and change colour scheme
        
        # plots peaks and troughs
        if plot_TSC and plot_peak_finder:
            plt.scatter(peaks_x1[channel_plot], peaks_y1[channel_plot], color = 'blue', s = 20)
            plt.scatter(troughs_x1[channel_plot], troughs_y1[channel_plot], color = 'brown', s = 20)
        # plots the chosen data range for analysis at each temperature hold
        if plot_TSC and plot_peak_finder:
            for temp in target_temperatures:
                peaks_x, peaks_y = result[(channel_plot,temp)]["Time: "], result[(channel_plot,temp)]["Temperature: "]
                plt.scatter(peaks_x, peaks_y, color = 'red', s = 20)

        if plot_TSC:
            plt.grid()
            plt.ylim(20,120)
            plt.xlabel("Time (s)")
            plt.ylabel("Temperature (degC)")
            plt.legend()
            plt.show()
    
results.to_csv("Complete analysis report.csv")

