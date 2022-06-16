## WWW
Follow the tutorial [here](tutorial) for the WWW task.

## Dependencies
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

## Usage
The following sections will work through how to use this code base to train and retrieve over the MSMARCO passage ranking data set.
## Training
You can download the train file `psg-train.tar.gz` for BERT from our resource [link](http://boston.lti.cs.cmu.edu/luyug/coil/msmarco-psg/). Alternatively, you can run pre-processing by yourself following the pre-processing [instructions](data_helpers).

Extract the training set from the tar ball and run the following code to launch training for msmarco passage.
```
python run_marco.py \  
  --output_dir $OUTDIR \  
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

## shaping documents & queries
You should make a new directory for the experiment.

You should have candidate documents in the form of
```
doc_id </t> document
```

NUMBER stands for the directory you made
```
# shaping documents
python scripts/www4_make_corpus_from_docs.py \
    --corpus_before ${www4 docs file path}  \
    --corpus_after ${NUMBER}/1_subset_corpus \
    --pid_file ${NUMBER}/pid_file \
    --dict ${NUMBER}/docid_to_intid_table.pkl
```
```
# shaping queries
python scripts/get_queries_from_topics.py \
    --topics_before ${WWW topics file} \
    --topics_after ${NUMBER}/1_shaped_www3_topics.txt
```

## splitting documents
Split documents to fit in BERT
```
INPUT=${NUMBER}/1_subset_corpus
OUTPUT=${NUMBER}/split_chunks
python scripts/document_truncation.py --input $INPUT --output $OUTPUT
```

## making corpus & queries for encoding
```
### make corpus for encoding
INPUT=${NUMBER}/split_chunks
SAVE_DIRECTORY=${NUMBER}/2_for_encoding_corpus
python data_helpers/common/encode_entry.py \
    --tokenizer_name bert-base-uncased \
    --truncate 512 \
    --input_file $INPUT \
    --save_to $SAVE_DIRECTORY

# make 100 splits
wc ${NUMBER}/2_for_encoding_corpus/split_chunks.json
# example↓ if there are 264755 lines
split -l 2648 -a 2 -d split_chunks.json split
```
```
### make queries for encoding
INPUT=${NUMBER}/1_shaped_www3_topics.txt
SAVE_DIRECTORY=${NUMBER}/2_for_encoding_queries
python data_helpers/common/encode_entry.py \
    --tokenizer_name bert-base-uncased \
    --truncate 16 \
    --input_file $INPUT \
    --save_to $SAVE_DIRECTORY
```

## making sure one document don't go to the other splits
```
CORPUS=${NUMBER}/2_for_encoding_corpus
for i in $(seq 0 98)
do
  j=$(($i+1))
  J=$(printf "%02d" $j)
  I=$(printf "%02d" $i)
  python scripts/split_corpus.py --split_before ${CORPUS}/split${I} --split_after ${CORPUS}/split${J}
done
```

## Encoding
After training, you can encode the corpus splits and queries.

You can download pre-processed data for BERT, `corpus.tar.gz, queries.{dev, eval}.small.json` [here](http://boston.lti.cs.cmu.edu/luyug/coil/msmarco-psg/). 
```
ENCODE_OUT_DIR=${NUMBER}/3_encoded_corpus_embeddings
CKPT_DIR=OUTDIR
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
If on a cluster, the encoding loop can be paralellized. For example, say if you are on a SLURM cluster, use `srun`,
```
for i in $(seq -f "%02g" 0 99)  
do  
  mkdir ${ENCODE_OUT_DIR}/split${i}  
  srun --ntasks=1 -c4 --mem=16000 -t0 --gres=gpu:1 python run_marco.py \  
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
    --encoded_save_path ${ENCODE_OUT_DIR}/split${i}&
done
```


Then encode the queries,
```
ENCODE_QRY_OUT_DIR=${NUMBER}/3_encoded_query_embeddings
CKPT_DIR=OUTDIR
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

## Retrieval
To use the fast retriever, you need to [compile the extension](retriever#fast-retriver). 

To do retrieval, run the following steps, 

(Note that there is no dependency in the for loop within each step, meaning that if you are on a cluster, you can distribute the jobs across nodes using `srun` or `qsub`.)

1) build document index shards
```
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
ENCODE_QRY_OUT_DIR=${NUMBER}/3_encoded_query_embeddings
QUERY_DIR=${NUMBER}/4_reformat_query
python retriever/format_query.py \  
  --dir $ENCODE_QRY_OUT_DIR \  
  --save_to $QUERY_DIR \  
  --as_torch
```

3) retrieve from each shard
```
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
SCORE_DIR=${NUMBER}/4_score
QUERY_DIR=${NUMBER}/4_reformat_query
python retriever/merger.py \  
  --score_dir ${SCORE_DIR}/intermediate/ \  
  --query_lookup  ${QUERY_DIR}/cls_ex_ids.pt \  
  --depth 1000 \  
  --save_ranking_to ${SCORE_DIR}/rank.txt

```

