#! /bin/bash

# $1 ../data/l1000/max10/
# $2 num of files, 5
# $3 train_max30.txt
# $4 l1000

echo $1$3
python preprocess.py --train $1""$3 --split 100 --jobs 16 --file_name l1000_mj_test
mkdir $1"pos"
mv $1""l1000_mj_test* $1"pos"

idx=0
while [ ${idx} -lt $2 ]; do
  echo $1${idx}"_neg.txt"
  #python test.py --split l1000
  mkdir $1${idx}"_neg"
  python preprocess.py --train $1${idx}"_neg.txt" --split 100 --jobs 16 --file_name l1000_mj_test
  mv $1""l1000_mj_test* $1${idx}"_neg"
  idx=$(( ${idx}+1 ))
done
