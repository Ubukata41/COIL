'''
split document into chunks
input should be 1_subset_corpus
'''

from argparse import ArgumentParser
import json
from io import open

parser = ArgumentParser()
parser.add_argument('--input', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()

MAX_INPUT_LENGTH = 510

# Text processing functions
def chunk_sent(sent, max_len):
    chunked_sents = []
    words = sent.strip().split()
    size = int(len(words) / max_len)
    for i in range(0, size):
        seq = words[i * max_len: (i + 1) * max_len]
        chunked_sents.append(' '.join(seq))
    return chunked_sents

with open(args.input) as input_corpus, open(args.output, 'w', encoding='utf-8') as output_corpus:
    i=0
    for line in input_corpus:
        docno, text = line.strip().split('\t')
        while(i==0):
            print(docno)
            i+=1
        # Split sentence if it's longer than BERT's maximum input length
        sentid=0
        if len(text) > MAX_INPUT_LENGTH:
            seq_list = chunk_sent(text, MAX_INPUT_LENGTH)
            for seq in seq_list:
                if len(str(sentid)) == 1:
                    sentno = docno + str("00") + str(sentid)
                elif len(str(sentid))==2:
                    sentno = docno + str("0") + str(sentid)
                else:
                    sentno = docno + str(sentid)
                output_corpus.write('{}\t{}\n'.format(sentno, seq))
                output_corpus.flush()
                sentid += 1
        else:
            sentno = docno + str("000")
            output_corpus.write('{}\t{}\n'.format(sentno, text))
            output_corpus.flush()
            sentid += 1