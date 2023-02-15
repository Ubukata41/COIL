from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--rank_before', required=True)
parser.add_argument('--rank_after', required=True)
args = parser.parse_args()

with open(args.rank_before) as input_f, open(args.rank_after, 'w', encoding='utf-8') as output_f:
        count = 0
        for line in input_f:
            queryid, docid, score = line.strip().split('\t')

            if docid[0] == "7":
                termid = "wb"
            elif docid[0] == "8":
                termid = "tw"
            elif docid[0] == "9":
                termid = "wt"

            if count < 1000:
                count += 1
            else:
                count = 1
                
            docid = 'clueweb12-' + docid[1:5] + termid + '-' + docid[5:7] + '-' + docid[7:12]

            output_f.write('{} {} {} {} {} {}\n'.format(queryid, "Q0", docid, count, score, "COIL"))