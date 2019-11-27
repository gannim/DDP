HOME=/data1/users/annie
PYTHON=$HOME/.venv/bin/python
WORK_HOME=$HOME/practice/DDP

## inner directory, DDP github 

WORK_PATH=$WORK_HOME/Bert

function download_bert() {
    proxy_on
    pushd $WORK_PATH
        git clone https://github.com/google-research/bert.git
        pushd bert
            mkdir pretrain_model
            pushd pretrain_model
                wget https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip
                unzip multi_cased_L-12_H-768_A-12.zip
            popd
        popd
    popd
    proxy_off
}

function download_dataset(){
    proxy_on
    pushd $WORK_PATH/classification/dataset
        git clone https://github.com/e9t/nsmc.git
        awk -F'\t' '{if($3 != "label") print $2"\t"$3 }' nsmc/ratings_train.txt > train.tsv
        awk -F'\t' '{if($3 != "label") print $2"\t"$3 }' nsmc/ratings_test.txt > dev.tsv
    popd
    proxy_off
}

download_bert
download_dataset

