'''
www形式のtopicファイルを整形する。整形後は
<query id><query>
の形式になる

usage
NUMBER=<your file>
python scripts/get_queries_from_topics.py \
    --topics_before ${path_to_topic_file} \
    --topics_after ${NUMBER}/1_shaped_www3_topics.txt
'''
import sys
import re
import ssl
from importlib import reload
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--topics_before', required=True)
parser.add_argument('--topics_after', required=True)
args = parser.parse_args()

def ntcir_topic_reader(topics_file):
    # Should be better to use xml reader instead?
    qid2query = {}
    qid = -1
    for line in topics_file:
        # Get topic number
        tag = 'qid'
        ind = line.find('<{}>'.format(tag))
        if ind >= 0:
            end_ind = -7
            qid = str(int(line[ind + len(tag) + 2:end_ind]))
        # Get topic title
        tag = 'content'
        ind = line.find('<{}>'.format(tag))
        if ind >= 0:
            end_ind = -11
            query = line[ind + len(tag) + 2:end_ind].strip()
            
            qid2query[qid] = query.replace('&apos;', "'").lower()

    return qid2query

with open(args.topics_before) as f, open(args.topics_after, 'w', encoding='utf-8') as out_file:
    a = ntcir_topic_reader(f)
    for key, value in a.items():
        out_file.write('{}\t{}\n'.format(key, value))
    