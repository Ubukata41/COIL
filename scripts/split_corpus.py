from argparse import ArgumentParser
import os
from ast import literal_eval

parser = ArgumentParser()
parser.add_argument('--split_before', required=True)
parser.add_argument('--split_after', required=True)
args = parser.parse_args()

with open(args.split_before, 'a') as before, open(args.split_after) as after:
    for i, line in enumerate(after):
        lst = literal_eval(line)
        if lst["pid"][-3:] == "000":
            break
        else:
            before.write("{}".format(line))

with open(args.split_after) as after, open("temp", 'w') as temp:
    for j, line in enumerate(after):
        if j < i:
            continue
        else:
            temp.write("{}".format(line))

os.rename("temp", args.split_after)