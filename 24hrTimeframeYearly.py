import csv
from datetime import datetime
import os

# Paths to input files
input_files = [
    'data/outputs/jan_apr_combined.csv',
    'data/outputs/may_aug_combined.csv',
    'data/outputs/sep_dec_combined.csv'
]

# Output file
output_file = 'data/outputs/24hrfilter.csv'

# Load all data
all_data = []

for file in input_files:
    with open(file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row['Timestamps'] = datetime.strptime(row['Timestamps'], "%Y-%m-%d %H:%M:%S")
                all_data.append(row)
            except Exception as e:
                print(f"Skipping invalid row in {file}: {e}")

# Group by date
daily_data = {}
for row in all_data:
    timestamp = row['Timestamps']
    date_key = timestamp.date()
    if timestamp.hour == 12 and timestamp.minute == 0 and timestamp.second == 0:
        daily_data[date_key] = row

# Sort by date and write to output
if not os.path.exists('data/outputs'):
    os.makedirs('data/outputs')

with open(output_file, 'w', newline='') as f:
    if daily_data:
        fieldnames = ['Timestamps'] + [k for k in next(iter(daily_data.values())).keys() if k != 'Timestamps']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for date in sorted(daily_data.keys()):
            row = daily_data[date]
            writer.writerow({**{'Timestamps': row['Timestamps'].strftime('%Y-%m-%d %H:%M:%S')}, **{k: row[k] for k in fieldnames if k != 'Timestamps'}})

print(f"Filtered data written to {output_file}")
