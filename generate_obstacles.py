WIDTH = 4096
HEIGHT = 256

f = open(f"obstacles_{WIDTH}x{HEIGHT}.dat", "w")

for i in range(WIDTH):
    f.write(f"{i} 0 1\n")
for i in range(WIDTH):
    f.write(f"{i} {HEIGHT - 1} 1\n")
for i in range(HEIGHT):
    f.write(f"{WIDTH - 1} {i} 1\n")
for i in range(HEIGHT):
    f.write(f"0 {i} 1\n")

## OPTIONAL
for i in range(HEIGHT):
    f.write(f"{WIDTH // 4} {i} 1\n")
for i in range(HEIGHT):
    f.write(f"{(WIDTH // 4) * 2} {i} 1\n")

f.close()