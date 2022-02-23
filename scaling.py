from lib2to3.pytree import convert


def get_times_lines():
    small_times_lines = []
    small_2_times_lines = []
    medium_times_lines = []
    big_times_lines = []
    for thread in range(1, 28):
        with open(f"d2q9-bgk_threads/d2q9-bgk_{thread}.out") as file:
            content = file.readlines()
            small_times_lines.append(content[11].strip())
            small_2_times_lines.append(content[17].strip())
            medium_times_lines.append(content[23].strip())
            big_times_lines.append(content[29].strip())
    return small_times_lines, small_2_times_lines, medium_times_lines, big_times_lines

def convert_to_float(times_lines):
    times = []
    for line in times_lines:
        time_with_unit = line.split("\t")[-1]
        time = float(time_with_unit.split(" ")[0])
        times.append(time)
    return times

def print_scalings(times):
    smallest_time = times[0]
    for time in times:
        print(smallest_time / time)

def main():
    small_times_lines, small_2_times_lines, medium_times_lines, big_times_lines = get_times_lines()
    small_times = convert_to_float(small_times_lines)
    small_2_times = convert_to_float(small_2_times_lines)
    medium_times = convert_to_float(medium_times_lines)
    big_times = convert_to_float(big_times_lines)
    # print_scalings(small_times)
    # print_scalings(small_2_times)
    # print_scalings(medium_times)
    # print_scalings(big_times)


if __name__ == "__main__":
    main()