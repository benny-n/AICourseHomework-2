import csv
import random

row_count = random.randrange(5, 11, 1)
col_count = random.randrange(5, 11, 1)
print(row_count)
print(col_count)
players_rows_loc = random.sample(population=[i for i in range(0, row_count)], k=2)
players_col_loc = random.sample(population=[i for i in range(0, col_count)], k=2)
print(players_rows_loc)
print(players_col_loc)
with open('boards/board2.csv','w',newline='') as f:
    writer = csv.writer(f, delimiter=' ')
    for i in range(0, row_count):
        row = random.choices(population=[0, -1], weights=[2, 1], k=col_count)
        if i in players_rows_loc:
            index = players_rows_loc.index(i)
            row[players_col_loc[index]] = index+1

        writer.writerow(row)
