num_labels = """2 1 3 2 8 2 1 1 5 6 4 9 9 5 4 6 3 4 2 6 3 1 5 6 3 0 3 6 8 8 3 2 4 9 2 0 4
 0 6 0 5 4 0 0 8 8 2 7 2 2 4 6 2 0 5 8 1 8 1 6 7 0 1 1 8 2 4 7 9 9 3 8 8 0
 1 5 1 5 6 3 8 0 1 0 8 1 1 2 3 7 4 1 9 6 7 1 9 2 5 0""".split()

text_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

for item in num_labels:
    item = int(item)
    print(item, text_labels[item])