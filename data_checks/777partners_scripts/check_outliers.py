#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
##                                                                 ##
##  +-----------------------------------------------------------+  ##
##  | Data Assessment                                           |  ##
##  +-----------------------------------------------------------+  ##
##                                                                 ##
##  Copyright (C) 2022 by Sion Cropper                             ##
##                                                                 ##
##  Contact Simon Cropper (simon.p.cropper@gmail.com) for more     ##
##  information or a demonstration.                                ##
##                                                                 ##
##  Function to execute all data assessment checks                 ##
##                                                                 ##
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#--------------------------------------------------------#
# Import required libraries
#--------------------------------------------------------#
import sys
import pandas as pd

#--------------------------------------------------------#
# Unpack arguments passed by main script
#--------------------------------------------------------#
# script_name = sys.argv[0]
# data_path = sys.argv[1]
flight_data_filename = sys.argv[2]
flight_gross_fare = sys.argv[3]
# col_2 = sys.argv[4]
# data_filename_2 = sys.argv[5]
perc_outlier_limit = int(float(sys.argv[6]))
# value_2 = sys.argv[7]

#--------------------------------------------------------#
# Load the data file
#--------------------------------------------------------#
try:
    df = pd.read_csv(flight_data_filename)
except IOError as e:
    print(f'FAIL, {e}')
    sys.exit()

#--------------------------------------------------------#
# Calculate boundaries to test for outlier values
# Out count number of outliers
#--------------------------------------------------------#
q1 = df[flight_gross_fare].quantile(0.25)
q3 = df[flight_gross_fare].quantile(.75)
iqr = q3 - q1
lower_boundary = q1 - (1.5 * iqr)
upper_boundary = q3 + (1.5 * iqr)
number_of_outliers = df[(df[flight_gross_fare] <= lower_boundary) | (df[flight_gross_fare] >= upper_boundary)].shape[0] # calculate # outliers
max_number_of_outliers = int(df.shape[0] * (perc_outlier_limit / 100)) # calculate max number of outliers

#--------------------------------------------------------#
# Return result
#--------------------------------------------------------#
print('PASS') if number_of_outliers < max_number_of_outliers else \
    print(f'FAIL, Number of outliers found: {number_of_outliers} : Max Number of Outliers allowed: {max_number_of_outliers}')