import csv
with open("trainingDataLot.csv") as in_file:
    with open("validationData.csv", 'w') as out_file:
        writer = csv.writer(out_file)
        for row in csv.reader(in_file):
            if row:
                writer.writerow(row)