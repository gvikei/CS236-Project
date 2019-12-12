num_labels = """1 0 1 6 4 9 6 2 7 2 9 2 2 4 2 9 7 7 2 2 1 7 5 6 3 0 3 8 0 2 0 1 1 0 7 5 3
 8 6 4 7 9 3 7 1 4 4 8 2 5 9 9 5 7 6 1 6 0 3 7 7 0 9 7 0 7 2 2 7 1 6 0 8 4
 4 8 2 3 1 4 1 3 9 7 4 3 2 3 0 9 8 0 0 7 9 1 6 7 4 4""".split()

text_labels = ["", "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

i = 0
for item in num_labels:
    if i % 8 == 0:
        i = 8
        print()
    item = int(item)
    print(text_labels[item], end='\t')
    i += 1