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
import os
import pandas as pd

#--------------------------------------------------------#
# Unpack arguments passed by main script
#--------------------------------------------------------#
# script_name = sys.argv[0]
data_path = sys.argv[1]
flight_data_filename = sys.argv[2]
flight_code = sys.argv[3]
valid_flight_code_col = sys.argv[4]
valid_flight_code_data_filename = sys.argv[5]
# value_1 = sys.argv[6]
# value_2 = sys.argv[7]

#--------------------------------------------------------#
# Load the data file
#--------------------------------------------------------#
try:
    df_flight_data = pd.read_csv(flight_data_filename)
except IOError as e:
    print(f'FAIL, {e}')
    sys.exit()

#--------------------------------------------------------#
# Load the valid flight code data file
#--------------------------------------------------------#
valid_flight_code_data_path_file_name = os.path.join(data_path, valid_flight_code_data_filename)
try:
    df_valid_flight_codes = pd.read_csv(valid_flight_code_data_path_file_name)
except IOError as e:
    print(f'FAIL, {e}')
    sys.exit()

#--------------------------------------------------------#
# Identify any invalid flight codes
#--------------------------------------------------------#
flight_codes_set = set(df_flight_data[flight_code])
valid_flight_code_set = set(df_valid_flight_codes[valid_flight_code_col])
invalid_flight_codes = list(flight_codes_set.difference(valid_flight_code_set))

#--------------------------------------------------------#
# Return result
#--------------------------------------------------------#
print('PASS') if not invalid_flight_codes else \
    print(f'FAIL, First 3 flight codes in error: {invalid_flight_codes[0:3]}, Number fight codes in error: {len(invalid_flight_codes)}, Number data rows: {df_flight_data.shape[0]}')