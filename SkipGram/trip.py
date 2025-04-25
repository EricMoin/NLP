lines = []
with open("result.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
values = []
for line in lines:
    divided = line.strip().split()
    if divided and divided[-1] != "0.000000":
        values.append(divided)
values.sort(key=lambda x: float(x[-1]), reverse=True)
with open("result_filtered.txt", "w", encoding="utf-8") as f:
    for value in values:
        f.write(" ".join(value) + "\n")
