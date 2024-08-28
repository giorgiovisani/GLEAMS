import re
import json
import numpy as np
import glob
from tqdm import tqdm
import os
import pandas as pd


DIR_NAME = os.path.dirname(__file__)

# Function to compute the 5th and 95th percentiles
def compute_percentiles(values):
    lower_percentile = np.percentile(values, 5)
    upper_percentile = np.percentile(values, 95)
    mean = np.mean(values)
    median = np.median(values)
    return lower_percentile, upper_percentile, mean, median

def compute_kpis():
    full_trip_filenames = glob.glob(os.path.join(DIR_NAME, 'results') + r"\*.json")
    results = []  # To store the results for DataFrame

    for file in tqdm(full_trip_filenames):
        # Load JSON data from file
        with open(file, 'r') as file:
            data = json.load(file)

        filename = os.path.basename(file.name)
        match = re.match(r"(local_mon|global_mon|recall)_(.*?)_sob(\d+)_(.*?)\.json", filename)
        if match:
            test_name, dataset, n_sobol_points, model = match.groups()

        # Compute and print percentiles for each method
        percentiles = {}
        for method, values in data.items():
            lower, upper, mean, median = compute_percentiles(values)
            results.append({
                'Test Name': test_name,
                'Dataset': dataset,
                'Sobol Points': n_sobol_points,
                'Model': model,
                'Method': method,
                '5th_percentile': lower,
                '95th_percentile': upper,
                'mean': mean,
                'median': median
            })

    # Convert results to a DataFrame
    df = pd.DataFrame(results)

    # Save the DataFrame to an Excel file
    output_file = os.path.join(DIR_NAME, 'results', 'percentiles_results.xlsx')
    df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == '__main__':
    # Run the function
    compute_kpis()