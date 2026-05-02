import csv

input_file = "output.csv"   # change this to your file name
output_file = "output.txt"

with open(input_file, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)

    # Get all column names
    fieldnames = reader.fieldnames

    filtered_rows = [row for row in reader if row["status"].strip().lower() == "escalated"]

with open(output_file, "w", encoding="utf-8") as txtfile:
    # Write header
    txtfile.write("\t".join(fieldnames) + "\n")

    # Write rows
    for row in filtered_rows:
        txtfile.write("\t".join(row[col] for col in fieldnames) + "\n")

print(f"Saved {len(filtered_rows)} escalated rows to {output_file}")