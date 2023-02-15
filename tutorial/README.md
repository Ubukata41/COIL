# Introduction
This code reproduces the work done in the We Want Web 4 Task.
The model is fine-tuned by MSMARCO passage dataset, performed retrieval on the WWW4 dataset.

# Dependencies
The code has been tested with,
```
pytorch==1.8.1
transformers==4.2.1
datasets==1.1.3
```
To use the retriever, you need in addition,
```
torch_scatter==2.0.6
faiss==1.7.0
```

# Training
You can download the train file `psg-train.tar.gz` for BERT from our resource [link](http://boston.lti.cs.cmu.edu/luyug/coil/msmarco-psg/). Alternatively, you can run pre-processing by yourself following the pre-processing [instructions](data_helpers).

Extract the training set from the tar ball and run the following code to launch training for msmarco passage.
```
python run_marco.py \  
  --output_dir MS_OUTDIR \  
  --model_name_or_path bert-base-uncased \  
  --do_train \  
  --save_steps 4000 \  
  --train_dir /path/to/psg-train \  
  --q_max_len 16 \  
  --p_max_len 128 \  
  --fp16 \  
  --per_device_train_batch_size 8 \  
  --train_group_size 8 \  
  --cls_dim 768 \  
  --token_dim 32 \  
  --warmup_ratio 0.1 \  
  --learning_rate 5e-6 \  
  --num_train_epochs 5 \  
  --overwrite_output_dir \  
  --dataloader_num_workers 16 \  
  --no_sep \  
  --pooling max 
```

# Shaping Documents & Queries
You should make a new directory for the experiment.

You should have candidate documents in the form of
```
doc_id </t> document
```

### Shaping documents
NUMBER stands for the directory you made.
```
NUMBER=<make directory yourself>
python scripts/www4_make_corpus_from_docs.py \
    --corpus_before ${path_to_www4_docs_file} \
    --corpus_after ${NUMBER}/1_subset_corpus \
    --pid_file ${NUMBER}/pid_file \
    --dict ${NUMBER}/docid_to_intid_table.pkl
```
### Shaping Queries
```
NUMBER=<>
python scripts/get_queries_from_topics.py \
    --topics_before ${path_to_topic_file} \
    --topics_after ${NUMBER}/1_shaped_www3_topics.txt
```

# Tokenize Corpus & Queries
### Tokenize Corpus for Encoding
Choose one of the two below steps.
- Split documents
This step splits long documents to fit in BERT.
```
bash scripts/split.sh ${NUMBER}
```
-# Original COIL
This step truncate the documents to the limit length of BERT.
```
bash scripts/original.sh ${NUMBER}
```

### Make Queries for Encoding
```
NUMBER=<>
INPUT=${NUMBER}/1_shaped_www3_topics.txt
SAVE_DIRECTORY=${NUMBER}/2_for_encoding_queries
python data_helpers/common/encode_entry.py \
    --tokenizer_name bert-base-uncased \
    --truncate 16 \
    --input_file $INPUT \
    --save_to $SAVE_DIRECTORY
```

# Encoding Corpus & Queries
After training, you can encode the corpus splits and queries.
### Encode Corpus
```
NUMBER=<>
ENCODE_OUT_DIR=${NUMBER}/3_encoded_corpus_embeddings
CKPT_DIR=MS_OUTDIR
CORPUS=${NUMBER}/2_for_encoding_corpus
mkdir $ENCODE_OUT_DIR
for i in $(seq -f "%02g" 0 99)  
do  
  mkdir ${ENCODE_OUT_DIR}/split${i}  
  python run_marco.py \  
    --output_dir $ENCODE_OUT_DIR \  
    --model_name_or_path $CKPT_DIR \  
    --tokenizer_name bert-base-uncased \  
    --cls_dim 768 \  
    --token_dim 32 \  
    --do_encode \  
    --no_sep \  
    --p_max_len 128 \  
    --pooling max \  
    --fp16 \  
    --per_device_eval_batch_size 128 \  
    --dataloader_num_workers 12 \  
    --encode_in_path ${TOKENIZED_DIR}/split${i} \  
    --encoded_save_path ${ENCODE_OUT_DIR}/split${i}
done
```

### Encode Queries
```
NUMBER=<>
ENCODE_QRY_OUT_DIR=${NUMBER}/3_encoded_query_embeddings
CKPT_DIR=MS_OUTDIR
TOKENIZED_QRY_PATH=${NUMBER}/2_for_encoding_queries/1_shaped_www3_topics.txt.json
python run_marco.py \  
  --output_dir $ENCODE_QRY_OUT_DIR \  
  --model_name_or_path $CKPT_DIR \  
  --tokenizer_name bert-base-uncased \  
  --cls_dim 768 \  
  --token_dim 32 \  
  --do_encode \  
  --p_max_len 16 \  
  --fp16 \  
  --no_sep \  
  --pooling max \  
  --per_device_eval_batch_size 128 \  
  --dataloader_num_workers 12 \  
  --encode_in_path $TOKENIZED_QRY_PATH \  
  --encoded_save_path $ENCODE_QRY_OUT_DIR
```
Note that here `p_max_len` always controls the maximum length of the encoded text, regardless of the input type.

# Retrieval
To use the fast retriever, you need to [compile the extension](retriever#fast-retriver). 

To do retrieval, run the following steps, 

(Note that there is no dependency in the for loop within each step, meaning that if you are on a cluster, you can distribute the jobs across nodes using `srun` or `qsub`.)

1) build document index shards
```
NUMBER=<>
ENCODE_OUT_DIR=${NUMBER}/3_encoded_corpus_embeddings
INDEX_DIR=${NUMBER}/4_index
for i in $(seq 0 9)  
do  
 python retriever/sharding.py \  
   --n_shards 10 \  
   --shard_id $i \  
   --dir $ENCODE_OUT_DIR \  
   --save_to $INDEX_DIR \  
   --use_torch
done  
```
2) reformat encoded query
```
NUMBER=<>
ENCODE_QRY_OUT_DIR=${NUMBER}/3_encoded_query_embeddings
QUERY_DIR=${NUMBER}/4_reformat_query
python retriever/format_query.py \  
  --dir $ENCODE_QRY_OUT_DIR \  
  --save_to $QUERY_DIR \  
  --as_torch
```

3) retrieve from each shard
```
NUMBER=<>
QUERY_DIR=${NUMBER}/4_reformat_query
INDEX_DIR=${NUMBER}/4_index
SCORE_DIR=${NUMBER}/4_score
for i in $(seq -f "%02g" 0 9)  
do  
  python retriever/retriever-fast.py \  
      --query $QUERY_DIR \  
      --doc_shard $INDEX_DIR/shard_${i} \  
      --top 1000 \
      --batch_size 512 \
      --save_to ${SCORE_DIR}/intermediate
      --shard_id $i
done 
```
4) merge scores from all shards
```
NUMBER=<>
SCORE_DIR=${NUMBER}/4_score
QUERY_DIR=${NUMBER}/4_reformat_query
python retriever/merger.py \  
  --score_dir ${SCORE_DIR}/intermediate/ \  
  --query_lookup  ${QUERY_DIR}/cls_ex_ids.pt \  
  --depth 1000 \  
  --save_ranking_to ${SCORE_DIR}/rank.txt

```

# Turn scores into WWW format
```
NUMBER=<>
SCORE_DIR=${NUMBER}/4_score
python scripts/score_to_www4_format.py 
  --rank_before ${SCORE_DIR}/rank.txt 
  --rank_after ${SCORE_DIR}/www_rank.txt 
  --dict ${NUMBER}/docid_to_intid_table.pkl
```

You can run evaluation using the tool here. [NTCIREVAL](http://research.nii.ac.jp/ntcir/tools/ntcireval-ja.html)
