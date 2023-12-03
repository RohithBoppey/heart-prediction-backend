import csv
import json
import requests

def test_model_with_csv(csv_file_path, api_url):
    # Read the CSV file
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)  # Assuming the first row is the header

        # Initialize counters for matching and total entries
        total_entries = 0
        matching_entries = 0

        for row in csv_reader:
            # Convert the CSV row to a dictionary
            row_dict = {header[i]: row[i] for i in range(len(header))}

            # Convert numeric values to the correct data type
            for key in row_dict:
                if key not in ['diabetes', 'smoking']:
                    row_dict[key] = float(row_dict[key])

            # Convert 'diabetes' and 'smoking' to boolean
            row_dict['diabetes'] = bool(int(row_dict['diabetes']))
            row_dict['smoking'] = bool(int(row_dict['smoking']))

            # Make a request to the Flask API for prediction
            response = requests.post(api_url + '/predict', json=row_dict)

            if response.status_code == 200:
                result = response.json()
                total_entries += 1

                # Check if the prediction matches the actual value in the CSV
                if result['prediction'] == int(row_dict['DEATH_EVENT']):
                    matching_entries += 1

    # Print the results
    print(f'Total entries: {total_entries}')
    print(f'Matching entries: {matching_entries}')
    print(f'Accuracy: {matching_entries / total_entries * 100:.2f}%')

# Assuming your CSV has a column named 'target_column' for the actual values
csv_file_path = '../Data/heart_failure_clinical_records_dataset orig.csv'
api_url = 'http://localhost:5000'  # Change this to your Flask app's URL
test_model_with_csv(csv_file_path, api_url)
