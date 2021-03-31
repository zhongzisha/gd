

with open('E:/Downloads/jpeg-9d/commands.txt', 'r') as fp:
    lines = [line.strip().split(' ') for line in fp.readlines()]

for line in lines:
    line[0] = 'cl'

lines = [' '.join(line) + '\n' for line in lines]
print(lines)

with open('E:/Downloads/jpeg-9d/compile.bat', 'w') as fp:
    fp.writelines(lines)