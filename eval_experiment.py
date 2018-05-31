import os
import re
import sys

import pandas as pd
import numpy as np

def get_epoch_num(epoch_string, suppress = True):
    try:
        results_list = re.findall("(?<=\[).*?(?=\])", epoch_string)
        epoch_num = float(results_list[0])
    except:
        if not(suppress):
            print("Epoch num failed: " + str(epoch_string))
        epoch_num = np.nan
    return epoch_num

def get_minibatch_num(epoch_string, suppress = True):
    try:
        results_list = re.findall("(?<=\[).*?(?=\])", epoch_string)
        minibatch_num = results_list[1]
    except:
        if not(suppress):
            print("Minibatch num failed: " + str(epoch_string))
        minibatch_num = np.nan
    return minibatch_num

def get_lr(epoch_string, suppress = True):
    try:
        results_list = epoch_string.split("lr: ")
        lr = float(results_list[-1])
    except:
        if not(suppress):
            print("lr failed: " + str(epoch_string))
        lr = np.nan
    return lr

def parse_train_string_n(in_string, n, suppress = True):
    try:
        results_list = in_string.split("(")
        train_string = results_list[0]
        proc_train_string = train_string[n:]
        final_train_string = float(proc_train_string[:-1])
    except:
        if not(suppress):
            print("Train string n failed: " + str(in_string))
        final_train_string = np.nan
    return final_train_string

def parse_val_string(in_string, suppress = True):
    try:
        results_list = in_string.split("(")
        val_string = results_list[1]
        val_string = float(val_string[:-1])
    except:
        if not(suppress):
            print("Val string n failed: " + str(in_string))
        val_string = np.nan
    return val_string

def preparse_result_df(file_name, suppress_val = True):
    result_tab = pd.read_table(file_name, header = None, names = ["Epoch_String", "Time_String", "Data_String", "Loss_String", "Prec1_String", "Prec5_String"])

    result_tab["Epoch"] = result_tab["Epoch_String"].apply(lambda epoch_string: get_epoch_num(epoch_string, suppress=suppress_val))
    result_tab["Minibatch"] = result_tab["Epoch_String"].apply(lambda epoch_string: get_minibatch_num(epoch_string, suppress=suppress_val))
    result_tab["lr"] = result_tab["Epoch_String"].apply(lambda epoch_string: get_lr(epoch_string, suppress=suppress_val))
    result_tab["Time_Train"] = result_tab["Time_String"].apply(lambda time_string: parse_train_string_n(time_string, 5, suppress = suppress_val))
    result_tab["Time_Avg"] = result_tab["Time_String"].apply(lambda time_string: parse_val_string(time_string, suppress = suppress_val))
    result_tab["Data_Train"] = result_tab["Data_String"].apply(lambda data_string: parse_train_string_n(data_string, 5, suppress = suppress_val))
    result_tab["Data_Avg"] = result_tab["Data_String"].apply(lambda data_string: parse_val_string(data_string, suppress = suppress_val))
    result_tab["Loss_Train"] = result_tab["Loss_String"].apply(lambda loss_string: parse_train_string_n(loss_string, 5, suppress = suppress_val))
    result_tab["Loss_Avg"] = result_tab["Loss_String"].apply(lambda loss_string: parse_val_string(loss_string, suppress = suppress_val))
    result_tab["Prec1_Train"] = result_tab["Prec1_String"].apply(lambda prec1_string: parse_train_string_n(prec1_string, 7, suppress = suppress_val))
    result_tab["Prec1_Avg"] = result_tab["Prec1_String"].apply(lambda prec1_string: parse_val_string(prec1_string, suppress = suppress_val))
    result_tab["Prec5_Train"] = result_tab["Prec5_String"].apply(lambda prec5_string: parse_train_string_n(prec5_string, 7, suppress = suppress_val))
    result_tab["Prec5_Avg"] = result_tab["Prec5_String"].apply(lambda prec5_string: parse_val_string(prec5_string, suppress = suppress_val))

    return result_tab

def postprocess_result_df(result_df, parsed_columns_only = False):
    final_indices = result_df["Epoch_String"].str.contains("Epoch: ")
    final_indices = final_indices.fillna(False)
    
    if (parsed_columns_only):
        final_columns = ["Epoch", "Minibatch", "lr", "Time_Train", "Time_Val", "Data_Train", "Data_Val", "Loss_Train", "Loss_Val", "Prec1_Train", "Prec1_Val", "Prec5_Train", "Prec5_Val"]
        result_df = result_df[final_columns]
    
    return result_df[final_indices]

def parse_test_df_row(row):
    epoch_string = row["Epoch_String"]
    list1 = epoch_string.split("Testing Results: Prec@1 ")
    list2 = list1[1].split(" Prec@5 ")
    prec1 = list2[0]
    list3 = list2[1].split(" Loss ")
    prec5 = list3[0]
    loss = list3[1]
    
    row["Prec1"] = float(prec1)
    row["Prec5"] = float(prec5)
    row["Loss"] = float(loss)
    
    return row
    
def parse_test_df(result_tab):
    test_indices = result_tab["Epoch_String"].str.contains("Testing Results")
    test_indices = test_indices.fillna(False)

    test_df_initial_columns = ["Epoch_String", "Time_String", "Data_String", "Loss_String", "Prec1_String"]
    test_df = result_tab[test_indices]
    test_df = test_df[test_df_initial_columns]
    test_df["Epoch"] = test_df.index.map(lambda index: result_tab.loc[index + 2, "Epoch"])
    
    final_columns = ["Epoch_String", "Epoch", "Prec1", "Prec5", "Loss"]
    test_df = test_df.apply(parse_test_df_row, axis = 1)
    test_df = test_df[final_columns]
    return test_df


if __name__ == '__main__':

    exp_name = sys.argv[1]
    if '/' in exp_name:
        # assume that this gives exp_name/timestamp
        exp_name, timestamp = exp_name.plit('/')
        print(f'Using the specified timestamp: {timestamp}')
    else:
        subdirs = os.listdir(os.path.join('log', exp_name))
        timestamp = sorted(subdirs)[-1] # latest timestamp
        print(f'Using the latest timestamp: {timestamp}')

    log_dir = os.path.join('log', exp_name, timestamp)

    file_name = [f for f in os.listdir(log_dir) if f != 'experiment_info.txt'][0]
    file_path = os.path.join(log_dir, file_name)

    results_path = os.path.join(log_dir, 'results.txt')

    result_df = preparse_result_df(file_path)
    test_df = parse_test_df(result_df)
    result_df = postprocess_result_df(result_df)
    # print(result_df.head())
    # print(test_df.head())
    with open(results_path, 'w') as f:
        f.write(f'Best Train Loss:\n----------------\n{result_df.loc[result_df.Loss_Train.idxmin()]}\n\n') 
        f.write(f'Best Train Prec@5:\n------------------\n{result_df.loc[result_df.Prec5_Train.idxmax()]}\n\n') 
        f.write(f'Best Train Prec@1:\n------------------\n{result_df.loc[result_df.Prec1_Train.idxmax()]}\n\n') 
        f.write(f'Best Test Loss:\n---------------\n{test_df.loc[test_df.Loss.idxmin()]}\n\n') 
        f.write(f'Best Test Prec@5:\n-----------------\n{test_df.loc[test_df.Prec5.idxmax()]}\n\n') 
        f.write(f'Best Test Prec@1:\n-----------------\n{test_df.loc[test_df.Prec1.idxmax()]}\n\n')
