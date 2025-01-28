import sys

filename = sys.argv[1]

f = open(filename, "r")
lines = f.readlines()
for line in lines:
    if "AiM AF" in line or "AiM RD_MAC" in line or "AiM WR_BIAS" in line or "AiM MAC_ABK" in line or "AiM EOC" in line:
        print(line[:-1])