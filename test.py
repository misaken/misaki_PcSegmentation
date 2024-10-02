first_frame = 80
n_frame = 40
skip = 5
for i in range(first_frame, first_frame+n_frame+1-skip, skip):
    print(i, i+skip)

for i in range(skip, n_frame+1, skip):
    print(i)