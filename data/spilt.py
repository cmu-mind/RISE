import json
import math

parent_dir=""
file_name=""
rows_per_file = 250

# List of input file names
input_files = []

input_files.append(f"{parent_dir}/{file_name}.jsonl")

# Number of rows per output file


# Initialize an empty list to store all the rows
all_rows = []

# Read the content from the input files
for file_name in input_files:
    with open(file_name, 'r') as file:
        for line in file:
            row = json.loads(line)
            all_rows.append(row)

# Calculate the number of output files needed
total_rows = len(all_rows)
num_output_files = math.ceil(total_rows / rows_per_file)
# num_output_files=1
# Split the rows into smaller files
for i in range(num_output_files):
    start_index = i * rows_per_file
    end_index = min((i + 1) * rows_per_file, total_rows)
    
    output_file_name = f"{parent_dir}/{file_name}_{i}.jsonl"
    with open(output_file_name, 'w') as output_file:
        for row in all_rows[start_index:end_index]:
            json.dump(row, output_file)
            output_file.write('\n')

print(f"Split {total_rows} rows into {num_output_files} files.")