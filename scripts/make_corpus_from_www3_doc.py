'''
www3_doc.csvから、COILで処理しやすいように形式を整える
<document id><\t><document>
の形にする。

usage
NUMBER=<make directory yourself>
python scripts_ubu/make_corpus_from_www3_doc.py \
    --corpus_before www3_data/www3_doc.csv \
    --corpus_after ${NUMBER}/subset_corpus_doc \
    --pid_file ${NUMBER}/pid_file_doc
'''

from argparse import ArgumentParser
from collections import defaultdict
import operator

parser = ArgumentParser()
parser.add_argument('--corpus_before', required=True)
parser.add_argument('--corpus_after', required=True)
parser.add_argument('--pid_file', required=True)
args = parser.parse_args()

def eval_bm25(collection_file, output_fn, pid_file, topK=1000):
    #クエリidとdocument idのペアに対するスコア
    doc_score_dict = defaultdict(dict)
    #クエリidとdocument idのペアに対するラベル、全部0
    doc_label_dict = defaultdict(dict)
    #query id とdocument idのペア
    top_doc_dict = defaultdict(list)
    #通し番号とdocument idのペア
    sent_dict = {}
    #qnoとqidのペア、意味わからん、同じもののペア
    q_dict = {}
    # document id と document text のペア
    my_dict = {}

    with open(collection_file) as bF, open(output_fn, 'w', encoding='utf-8') as out_file, open(pid_file, 'w', encoding='utf-8') as pid_file:
        for line in bF:
            #dnoは通し番号0~875883, qid=qnoはクエリid, sidはdoc id, labelは全部0
            label, score, _, docs, qid, sid, qno, dno = line.strip().split('\t')
            sent_dict[dno] = sid
            q_dict[qno] = qid  

            my_dict[sid] = docs

        # 512トークンを超える文章のカウント
        over = 0
        less = 0
        total_length = 0

        # document id とdocumentを書く
        for key, value in my_dict.items():
            # wb, tw, wtを識別できるようにする
            # wbの場合7を、twの場合8を、wtの場合9をdocumentidの先頭に付ける
            if key[7:9] != "12":
                print(key[7:9])
            if key[14:16] == "wb":
                termid = "7"
            elif key[14:16] == "tw":
                termid = "8"
            elif key[14:16] == "wt":
                termid = "9"
            # documentidは数値である必要があるため、文字列を消す
            documentid = termid + key[10:14]+ key[17:19] + key[20:]
            # doc_pair.tsvはいわゆるコーパスのサブセット、pid_fileはdocumentidの羅列。ただの確認用
            out_file.write('{}\t{}\n'.format(documentid, value))
            pid_file.write('{}, id: {}\n'.format(key, documentid))
            if len(value.strip().split()) > 512:
                over += 1
            else:
                less += 1
            total_length += len(value.strip().split())

        intkey= []
        for key in q_dict.keys():
            intkey.append(int(key))

        pid_file.write('more than 512 tokens : {}\n'.format(over))
        pid_file.write('less than 512 tokens : {}\n'.format(less))
        pid_file.write('number of documents : {}\n'.format(over + less))
        pid_file.write('average length of documents : {}\n'.format(total_length / (over + less)))
        pid_file.write('query ids : {}\n\n'.format(q_dict.keys()))
        pid_file.write('sorted query ids : {}\n\n'.format(sorted(intkey)))
        pid_file.write('number of queries : {}\n'.format(len(q_dict)))


eval_bm25(args.corpus_before, args.corpus_after, args.pid_file)