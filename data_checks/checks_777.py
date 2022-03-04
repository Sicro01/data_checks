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
import subprocess
from tkinter.tix import MAIN
import pandas as pd
import sys
import os
import numpy as np
from datetime import datetime
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor
from functools import partial
# import time
from timeit import default_timer as timer
from datetime import timedelta
import logging
import warnings
warnings.filterwarnings('ignore')
#--------------------------------------------------------#
# Global Variables
#--------------------------------------------------------#
DATA_DIR = '777partners_data'
DATA_FOLDER_NAME = ''
SCRIPT_CHECKLIST_DIR = '777partners_checklist'
SCRIPT_CHECKLIST_FILE_NAME = ''
OUTPUT_DIR = '777partners_out'
LOG_DIR = '777partners_logs'
DEBUG_LEVEL = 'INFO'
SCRIPT_DIR = '777partners_scripts'
VALID_COMPANIES = ['flair', 'lifecents', '1190sports']
BATCH_SIZE = 50

#--------------------------------------------------------#
# Function: Initialise variables
#--------------------------------------------------------#
def init():
    # Declare the global variables in use in this function
    global OUTPUT_DIR, DATA_DIR, DATA_PATH, LOG_DIR, LOG_PATH_FILE_NAME, LOG_FORMATTER, SCRIPT_DIR, SCRIPT_PATH, SCRIPT_CHECKLIST_DIR, SCRIPT_CHECKLIST_PATH, \
        SCRIPT_CHECKLIST_FILE_NAME, OUTPUT_PATH, MAIN_LOGGER, DATA_FOLDER_NAME, SCRIPT_CHECKLIST_FILE_NAME, BATCH_SIZE

    # Clear users console
    os.system('cls' if os.name == 'nt' else 'clear') # Clear users screen before outputs messages begin

    get_inputs()

    # Create log & output dirs if they don't exist
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, DATA_FOLDER_NAME)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    LOG_DIR = os.path.join(LOG_DIR, DATA_FOLDER_NAME)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    # Set directory / name for scripts and checklist
    SCRIPT_CHECKLIST_DIR = os.path.join(SCRIPT_CHECKLIST_DIR, DATA_FOLDER_NAME)
    SCRIPT_CHECKLIST_FILE_NAME = f'{DATA_FOLDER_NAME}_checklist.csv'

    # Set path for all data files
    DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), DATA_DIR, DATA_FOLDER_NAME)
    SCRIPT_CHECKLIST_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), SCRIPT_CHECKLIST_DIR)
    OUTPUT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), OUTPUT_DIR)
    SCRIPT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), SCRIPT_DIR)
    
    # Load data checks
    df_checklist_full, data_file_name = load_checklist()
    print(data_file_name)
    # Set up main log - create the log dir if it doesn't exist
    timestamp = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
    log_filename = f'{DATA_FOLDER_NAME}_data_assessment_{data_file_name}_{timestamp}.log'
    LOG_PATH_FILE_NAME = os.path.join(os.path.dirname(os.path.realpath(__file__)), LOG_DIR, log_filename)
    LOG_FORMATTER = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - PID:%(process)d - %(message)s', '%Y-%m-%d %H:%M:%S')
    main_logger_name = f'777partners_data_assessment_{data_file_name}'
    MAIN_LOGGER = create_log(main_logger_name)

    # Log checklist loaded messages
    MAIN_LOGGER.info(f'Loaded: {SCRIPT_CHECKLIST_FILE_NAME}\n')
    MAIN_LOGGER.debug(f'Shape: {df_checklist_full.shape}')
    MAIN_LOGGER.debug(f'First row: {df_checklist_full.head(1)}\n')
    
    # Divide checks into batches
    list_checklist_batch_dfs = [df_checklist_full[i:i + BATCH_SIZE] for i in range(0, df_checklist_full.shape[0], BATCH_SIZE)]
    number_of_batches = len(list_checklist_batch_dfs)

    MAIN_LOGGER.info(f'Starting execution of {len(df_checklist_full):.0f} data assessment checks splitting {number_of_batches} into max batch size of {BATCH_SIZE} checks')
    return list_checklist_batch_dfs, number_of_batches, data_file_name

#--------------------------------------------------------#
# Function: Initialise variables
#--------------------------------------------------------#
def get_inputs():
    global VALID_COMPANIES, DATA_FOLDER_NAME
    # Get folder where data is stored
    while True:
        data_folder_name = str(input(f'Enter folder name, must be one of {VALID_COMPANIES}. Default is {VALID_COMPANIES[0]}:') or f'{VALID_COMPANIES[0]}')
        data_folder_name = data_folder_name.replace('\n', '').replace('\r', '')
        if data_folder_name not in VALID_COMPANIES:
            print("Sorry, that is not a valid company.")
            continue
        else:
            break
    DATA_FOLDER_NAME = data_folder_name

#--------------------------------------------------------#
# Create a log and it's handlers
#--------------------------------------------------------#
def create_log(name, script_log_file_path_name='', script_log_formatter=''):
    global LOG_PATH_FILE_NAME, LOG_FORMATTER, DEBUG_LEVEL

    LOG_PATH_FILE_NAME = script_log_file_path_name if script_log_file_path_name != '' else LOG_PATH_FILE_NAME
    LOG_FORMATTER = script_log_formatter if script_log_formatter != '' else LOG_FORMATTER

    logger = logging.getLogger(name)
    log_level = logging.getLevelName(DEBUG_LEVEL)
    logger.setLevel(log_level)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(LOG_FORMATTER)
    logger.addHandler(console_handler)

    # Create file handler
    file_handler = logging.FileHandler(LOG_PATH_FILE_NAME)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(LOG_FORMATTER)
    logger.addHandler(file_handler)
    return logger

#--------------------------------------------------------#
# Cleanly close down the log handler(s)
#--------------------------------------------------------#
def log_remove_handlers(log):
    log.info('CLOSE LOG: log_remove_handlers: Removing all existing log handlers')
    # get all loggers
    loggers = [logging.getLogger(name) if '777' in name else None for name in logging.root.manager.loggerDict]
    # for each valid logger remove all handlers
    for log in loggers:
        if log != None:
            while bool(len(log.handlers)):
                for handler in log.handlers:
                    log.removeHandler(handler)
    del log

#--------------------------------------------------------#
# Function: Load checklist file which lists all the 
# data checks to be performed
#--------------------------------------------------------#
def load_checklist():
    global SCRIPT_CHECKLIST_PATH, SCRIPT_CHECKLIST_FILE_NAME, MAIN_LOGGER
    try:
        checklist_path_name = os.path.join(SCRIPT_CHECKLIST_PATH, SCRIPT_CHECKLIST_FILE_NAME)
        df = pd.read_csv(checklist_path_name)
        df.columns = df.columns.str.strip().str.lower().str.replace(r'\s+', '_').str.replace('-', '_').str.replace('(', '').str.replace(')', '') \
            .str.replace('/', '_').str.replace('#', 'no').str.replace('.', '')
        
        # Get name of the data file to be processed
        data_file_name = df['data1'].values[0]

        # Filter on the checks set to ENABLE and replace NAN values with an empty string
        df = df[df['enable'] == 'Y']

        # Replace nan data in cells with an empty cell value
        df = df.replace(np.nan, '', regex=True) 

        # Remove any spurious space the checklist values may have
        df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        return df, data_file_name
    except Exception as e:
        print(f'Fatal Script Error: {__file__}\n')
        print(f'Error Message: {e}')
        # MAIN_LOGGER.critical(f'Fatal Script Error: {__file__}\n')
        MAIN_LOGGER.critical(f'Error Message: {e}')
        sys.exit()

#--------------------------------------------------------#
# Function: Save results
#--------------------------------------------------------#
def save_results(results, batch_number, data_file_name):
    global OUTPUT_PATH, LOG_FORMATTER, DATA_FOLDER_NAME, MAIN_LOGGER
    # Set up output file name and path
    timestamp = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
    output_file_name = f'{DATA_FOLDER_NAME}_data_assessment_results_datafile_{data_file_name}_batch_{batch_number+1}_{timestamp}.csv'
    output_path_name = os.path.join(OUTPUT_PATH, output_file_name)

    # Save results to csv
    results = list(results)
    MAIN_LOGGER.info(f'SAVING RESULTS: DATA FILE:{data_file_name} BATCH:{batch_number+1}: {len(results)} results.')
    columns = ['id', 'use_case', 'script', 'column1', 'data1', 'data2', 'value1', 'value2', 'pass_fail', 'result_value', 'error_message']
    df_results = pd.DataFrame(results, columns=columns)
    df_results.replace({'\n': ''}, regex=True, inplace=True) # Remove newline chars in result and error_message as this messes up the write to csv file
    df_results.to_csv(output_path_name, index=False)
    MAIN_LOGGER.info(f'SAVED RESULTS: File:{output_path_name}.')
    
    # Return pass/fails numbers
    num_passes = df_results[df_results['pass_fail'] == 'PASS'].shape[0]
    num_fails = df_results[df_results['pass_fail'] == 'FAIL'].shape[0]
    return num_passes, num_fails

#--------------------------------------------------------#
# Function: Output processing summary
#--------------------------------------------------------#
def close_batch(start_batch_time, num_passes, num_fails, num_checks, batch_number):
    global MAIN_LOGGER
    # Calculate run time and format it
    end_batch_time = timer()
    execution_time = timedelta(seconds=end_batch_time - start_batch_time)
    total_execution_seconds = execution_time.seconds
    hours, remainder = divmod(total_execution_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Final log entry for this batch
    MAIN_LOGGER.info(f'END OF EXECUTION OF BATCH {batch_number+1}: Duration: {hours:02d}:{minutes:02d}:{seconds:02d} - {num_checks:.0f}'
        f' data assessment checks executed in with {num_passes} passes and {num_fails} failures')

#--------------------------------------------------------#
# Function: Convert a data check from the checklist
# dataframe into a named tuple
#--------------------------------------------------------#
def convert_data_check_to_namedtuple(data_check):
    # Convert the tuple passed into a dictionary
    data_check_dict = {**data_check[1]} # Convert pandas series to a dict

    # Convert values passed to strings - scripts fails it you don't do this
    data_check_dict['value1'] = str(data_check_dict['value1'])
    data_check_dict['value2'] = str(data_check_dict['value2'])

    # Convert the tuple passed into a named tuple - so variable names read easier
    Datacheck = namedtuple('My_Tuple', data_check_dict) # Create namedtuple defn
    this_datacheck = Datacheck(**data_check_dict) # Convert dict to named tuple
    return this_datacheck

#--------------------------------------------------------#
# Function: Create and submit the subprocess to execute
# the python statement to run a data check
#--------------------------------------------------------#
def execute_data_check(script_path, data_path, data_check_logger, this_datacheck):
    # Build script and data paths for the underlying data check scripts
    script_path_file_name = os.path.join(script_path, this_datacheck.script)
    data1_data_path_file_name = os.path.join(data_path, this_datacheck.data1)
    data2_data_path_file_name = os.path.join(data_path, this_datacheck.data2)
    
    # Execute this data check script
    data_check_logger.info(f'Submitting script: Id:{this_datacheck.use_case_id} Script:{this_datacheck.script} Col:{this_datacheck.column1}')

    p1 = subprocess.run(['python', script_path_file_name, # arg 0
        data_path, # arg 1
        data1_data_path_file_name, # arg 2
        this_datacheck.column1, # arg 3
        this_datacheck.column2, # arg 4
        data2_data_path_file_name, # arg 5
        this_datacheck.value1, # arg 6
        this_datacheck.value2], # arg 7
        shell=True, capture_output=True, text=True)
    return p1

#--------------------------------------------------------#
# Function: Process and format the resul from a data check
#--------------------------------------------------------#
def process_data_check(this_datacheck, data_check_logger, p1):
    # Some scripts return both a PASS/FAIL result and a numeric value (e.g. col_skewness)
    # Those scripts return a comma seperated string e.g. "PASS, {col_skewness}" - so we split those two values out if there are two values
    p1_return_values = [return_value.strip() for return_value in p1.stdout.split(',', 1)]
    p1_pass_fail = p1_return_values[0]
    p1_result_value = p1_return_values[1] if len(p1_return_values) > 1 else ''

    # Set return information based on script return code
    if p1.returncode == 0: # Means script has worked - but may have passed or failed the actual data check
        if p1_pass_fail == 'PASS':
            error_message = 'DATA CHECK ISSUED PASS'
            data_check_logger.info(f'DATA CHECK ISSUED PASS: Id:{this_datacheck.use_case_id} Script:{this_datacheck.script}')
        else:
            error_message = 'DATA CHECK ISSUED FAIL'
            data_check_logger.info(f'DATA CHECK ISSUED FAIL: Id:{this_datacheck.use_case_id} Script:{this_datacheck.script}')
    else:  # Means script has failed with a fatal script error
        p1_pass_fail = 'FAIL'
        error_message = p1.stderr
        data_check_logger.info(f'FATAL SCRIPT ERROR: Id:{this_datacheck.use_case_id} Script:{this_datacheck.script}')
        data_check_logger.debug(f'FATAL SCRIPT ERROR: Id:{this_datacheck.use_case_id} Script:{this_datacheck.script} Error:{p1.stderr}')

    this_result = [this_datacheck.use_case_id, this_datacheck.use_case, this_datacheck.script, this_datacheck.column1, this_datacheck.data1, this_datacheck.data2, \
            this_datacheck.value1, this_datacheck.value2, p1_pass_fail, p1_result_value, error_message]
    return this_result

#--------------------------------------------------------#
# Function: Execute a data check 
#--------------------------------------------------------#
def data_check_subprocess(script_log_file_path_name, script_log_formatter, script_path, data_path, data_check):
    # Convert data check to named tuple
    this_datacheck = convert_data_check_to_namedtuple(data_check)

    # Create log for the data check
    data_check_logger_name = f'777partners_data_assessment_Id:{this_datacheck.use_case_id}_script:{this_datacheck.script}'
    data_check_logger = create_log(data_check_logger_name, script_log_file_path_name, script_log_formatter)
    
    # Submit the python process to execute the data check
    p1 = execute_data_check(script_path, data_path, data_check_logger, this_datacheck)
    
    # Process the result of the data check
    this_result = process_data_check(this_datacheck, data_check_logger, p1)

    # Cleanly close down the log handler(s) for the data check log
    log_remove_handlers(data_check_logger)

    return this_result

def main():
    global LOG_PATH_FILE_NAME, MAIN_LOGGER, LOG_FORMATTER, SCRIPT_PATH, DATA_PATH, BATCH_SIZE

    # Initialise
    list_checklist_batch_dfs, number_of_batches, data_file_name = init()
    
    # To help overcome memory issues process checks in batches
    for batch_number, df_checklist_batch in enumerate(list_checklist_batch_dfs):
        # Start timer
        start_batch_time = timer()

        # Execute all data checks using parallel processing
        MAIN_LOGGER.info(f'Starting - batch {batch_number+1} of {number_of_batches} batches\n')
        func = partial(data_check_subprocess, LOG_PATH_FILE_NAME, LOG_FORMATTER, SCRIPT_PATH, DATA_PATH) # We need this to pass multiple args to the data_check_subprocess function
        with ProcessPoolExecutor(max_workers=None) as executor:
            results = executor.map(func, df_checklist_batch.iterrows())
        MAIN_LOGGER.info(f'Ending - batch {batch_number+1} of {number_of_batches} batches\n')

        # Save results
        num_passes, num_fails = save_results(results, batch_number, data_file_name)
        
        # Output execution time and issue closing log message
        close_batch(start_batch_time, num_passes, num_fails, len(df_checklist_batch), batch_number)
        
    # Cleanly close down the main log handler(s)
    log_remove_handlers(MAIN_LOGGER)

#--------------------------------------------------------#
# Main Function: Execute Data Assessment
#--------------------------------------------------------#
if __name__ == "__main__":
    main()
        
