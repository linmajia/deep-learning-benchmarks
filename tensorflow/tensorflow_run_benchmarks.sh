#!/bin/bash

CMBS=4
CNB=1000

RMBS=32
RNB=1000

FMBS=128
FNB=1000

for i in "$@"
do
case $i in
    -cm=*|--cnn_mini_batch_size=*)
    CMBS="${i#*=}"
    shift # past argument=value
    ;;
    -cn=*|--cnn_num_batch=*)
    CNB="${i#*=}"
    shift # past argument=value
    ;;

    -rm=*|--rnn_mini_batch_size=*)
    RMBS="${i#*=}"
    shift # past argument=value
    ;;
    -rn=*|--rnn_num_batch=*)
    RNB="${i#*=}"
    shift # past argument=value
    ;;

    -fm=*|--fcn_mini_batch_size=*)
    fMBS="${i#*=}"
    shift # past argument=value
    ;;
    -fn=*|--fcn_num_batch=*)
    FNB="${i#*=}"
    shift # past argument=value
    ;;
	
    --default)
    DEFAULT=YES
    shift # past argument with no value
    ;;
    *)
            # unknown option
    ;;
esac
done

rm -f output_alexnet.log
rm -f output_resnet.log
rm -f output_fcn5.log
rm -f output_fcn8.log
rm -f output_lstm32.log
rm -f output_lstm64.log

CUDA_VISIBLE_DEVICES='0' python benchmark.py --arch fcn5 --batch-size ${FMBS} --num-batches ${FNB} 2>&1 | tee output_fcn5.log
CUDA_VISIBLE_DEVICES='0' python benchmark.py --arch fcn8 --batch-size ${FMBS} --num-batches ${FNB} 2>&1 | tee output_fcn8.log
CUDA_VISIBLE_DEVICES='0' python benchmark.py --arch alexnet --batch-size ${CMBS} --num-batches ${CNB} 2>&1 | tee output_alexnet.log
CUDA_VISIBLE_DEVICES='0' python benchmark.py --arch resnet --batch-size ${CMBS} --num-batches ${CNB} 2>&1 | tee output_resnet.log
CUDA_VISIBLE_DEVICES='0' python rnn/lstm/lstm.py --batchsize ${RMBS} --iters ${RNB} --seqlen 32 --numlayer 2 --hiddensize 256 --device 0 --data_path ../cntk/rnn/PennTreebank/Data 2>&1 | tee output_lstm32.log
CUDA_VISIBLE_DEVICES='0' python rnn/lstm/lstm.py --batchsize ${RMBS} --iters ${RNB} --seqlen 64 --numlayer 2 --hiddensize 256 --device 0 --data_path ../cntk/rnn/PennTreebank/Data 2>&1 | tee output_lstm64.log
 
