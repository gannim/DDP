HOME=${1:-""}
PYTHON=$HOME/.venv/bin/python
WORK_HOME=$HOME/practice/DDP

## inner directory, DDP github 

WORK_PATH=$WORK_HOME/Bert
GOOGLE_BERT_PATH=$WORK_PATH/bert
PRETRAIN_PATH=$GOOGLE_BERT_PATH/pretrain_model/multi_cased_L-12_H-768_A-12

FINE_TUNE_PATH=$WORK_PATH/kor-quad
DATASET_PATH=$FINE_TUNE_PATH/dataset

mkdir $FINE_TUNE_PATH/output

TRAIN_PATH=$DATASET_PATH/KorQuAD_v1.0_train.json
DEV_PATH=$DATASET_PATH/KorQuAD_v1.0_dev.json

CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON $GOOGLE_BERT_PATH/run_squad.py \
--do_train=true \
--do_eval=true \
--do_lower_case=false \
--vocab_file=$PRETRAIN_PATH/vocab.txt \
--train_file=$TRAIN_PATH --do_predict \
--predict_file=$DEV_PATH \
--init_checkpoint=$PRETRAIN_PATH/bert_model.ckpt \
--bert_config_file=$PRETRAIN_PATH/bert_config.json \
--output_dir=$FINE_TUNE_PATH/output \
--max_seq_length=512 \
--train_batch_size=32
