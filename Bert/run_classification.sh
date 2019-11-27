HOME=/data1/users/annie
PYTHON=$HOME/.venv/bin/python
WORK_HOME=$HOME/practice/DDP

## inner directory, DDP github 

WORK_PATH=$WORK_HOME/Bert
GOOGLE_BERT_PATH=$WORK_PATH/bert
PRETRAIN_PATH=$GOOGLE_BERT_PATH/pretrain_model/multi_cased_L-12_H-768_A-12

FINE_TUNE_PATH=$WORK_PATH/classfication
DATASET_PATH=$FINE_TUNE_PATH/dataset

mkdir $FINE_TUNE_PATH/output

# define binary classification task
cp $FINE_TUNE_PATH/run_classifier.py $GOOGLE_BERT_PATH/

CUDA_VISIBLE_DEVICES=1,2,3 $PYTHON $GOOGLE_BERT_PATH/run_classifier.py \
--task_name=bic \
--data_dir=$DATASET_PATH \
--output_dir=$FINE_TUNE_PATH/output \
--vocab_file=$PRETRAIN_PATH/vocab.txt \
--init_checkpoint=$PRETRAIN_PATH/bert_model.ckpt \
--bert_config_file=$PRETRAIN_PATH/bert_config.json \
--do_train=true \
--do_eval=true \
--do_lower_case=false \
--learning_rate=2e-5 \
--max_seq_length=128 \
--num_train_epochs=3.0 \
--train_batch_size=32

