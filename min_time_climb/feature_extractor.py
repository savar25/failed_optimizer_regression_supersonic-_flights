import tsfresh
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tsfresh import extract_features,extract_relevant_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
import itertools
from tsfresh.feature_extraction import ComprehensiveFCParameters



def generate_failures_csv(tf_data: pd.DataFrame, threshold_percentage: float):
    # Calculate absolute error for each time step per id
    tf_data['error'] = abs(tf_data['h_sim'] - tf_data['h'])
    
    # Compute mean error per id
    mean_errors = tf_data.groupby('id')['error'].mean().reset_index()
    
    
    # Determine threshold value (5% of mean h values)
    mean_h = tf_data.groupby('id')['h'].mean().reset_index()
    mean_errors = mean_errors.merge(mean_h, on='id')
    mean_errors['threshold'] = mean_errors['h'] * (threshold_percentage / 100)
    print(mean_errors)
    # Determine failure status
    mean_errors['failure'] = (mean_errors['error'] > mean_errors['threshold']).astype(int)
    
    # Select relevant columns
    failures = mean_errors[['id', 'failure']]
    
    # Save to CSV
    failures.to_csv('failures.csv', index=False,header=False)



num_cases = 50

all_data = []



for i in itertools.chain(range(1, num_cases+1)):
    

    fufile = os.path.join(os.path.dirname(__file__), f"Data_Files/data_case_{i}.csv")
    # tffile = os.path.abspath('./{} Result Data/{} Impact t-F Data.csv'.format(basename, basename))
    # ahfile = os.path.abspath('./{} Result Data/ Assembly History Outputs.csv'.format(basename, basename))

    with open(fufile, 'r') as f:
        tf = pd.read_csv(f)
        for idx, series in tf.iterrows():
            time, h, v,h_sim,v_sim = series
            all_data.append([i, time, h,h_sim])

tf_data = pd.DataFrame(all_data, columns=['id', 'time', 'h','h_sim'])
generate_failures_csv(tf_data,35)

custom_settings=ComprehensiveFCParameters()

custom_settings = {
    'mean': None,
    'standard_deviation': None,
    'variance': None,
    'maximum': None,
    'minimum': None,
    # 'absolute_sum_of_changes': None,
    # 'longest_strike_above_mean': None,
    # 'longest_strike_below_mean': None
}
extracted_features = extract_features(tf_data, column_id='id', column_sort='time', column_value="h" ,default_fc_parameters=custom_settings,n_jobs=0)
impute(extracted_features)





# print('Extracted {} Features'.format(extracted_features.shape))

# impute(extracted_features)

extracted_features.to_csv('extracted_features.csv', index=False)


results_info = 'failures.csv'
with open(results_info, 'r') as f:
    y1 = pd.read_csv(f, header=None)

y = pd.Series(data=y1[1])
y.index = y1[0]


relevant_features = select_features(extracted_features, y, n_jobs=0)
relevant_features.to_csv('extracted_features.csv', index=False)
print(relevant_features)
