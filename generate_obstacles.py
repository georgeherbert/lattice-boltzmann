def output(i):
    f = open(f"obstacles_{i}x{i}.dat", "w")
    for j in range(i):
        f.write(f"0 {j} 1\n")
    for j in range(i):
        f.write(f"{j} 0 1\n")
    for j in range(i):
        f.write(f"{i - 1} {j} 1\n")
    for j in range(i):
        f.write(f"{j} {i - 1} 1\n")

output(2048)