from argparse import ArgumentParser

import pickle

parser = ArgumentParser()
parser.add_argument('--rank_before', required=True)
parser.add_argument('--rank_after', required=True)
parser.add_argument('--dict', required=True)
args = parser.parse_args()

with open(args.dict, "rb") as tf:
    docid_to_intid_table = pickle.load(tf)

with open(args.rank_before) as input_f, open(args.rank_after, 'w', encoding='utf-8') as output_f:
        count = 0
        for line in input_f:
            queryid, docid, score = line.strip().split('\t')
            docid=int(docid)
            id = docid_to_intid_table[docid]

            if count < 1000:
                count += 1
            else:
                count = 1

            output_f.write('{} {} {} {} {} {}\n'.format(queryid, "Q0", id, count, score, "COIL"))