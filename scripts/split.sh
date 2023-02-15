#!/bin/bash

NUMBER=$1

# Split documents to fit in BERT
echo "Splitting Documents..."
python scripts/document_truncation.py --input "${NUMBER}/1_subset_corpus" --output "${NUMBER}/split_chunks"
echo "Tokenizing Documents..."
python data_helpers/common/encode_entry.py --tokenizer_name bert-base-uncased --truncate 512 --input_file "${NUMBER}/split_chunks" --save_to "${NUMBER}/2_for_encoding_corpus"
echo "Making splits for Encoding..."
lines=0
lines=$(cat ${NUMBER}/2_for_encoding_corpus/split_chunks.json | wc -l)
echo "lines: $lines"
cd ${NUMBER}/2_for_encoding_corpus

lines=$[lines/100 + 1]
split -l $lines -a 2 -d split_chunks.json split

cd ../../

# making sure one document don't go to the other splits
for i in $(seq 0 98)
do
  j=$(($i+1))
  J=$(printf "%02d" $j)
  I=$(printf "%02d" $i)
  python scripts/split_corpus.py --split_before "${NUMBER}/2_for_encoding_corpus/split${I}" --split_after "${NUMBER}/2_for_encoding_corpus/split${J}"
done

echo "Done"