# icml18_l1000

- 학습을 돌리기위한 최소의 data/file을 남겨놓음.

```
export PYTHONPATH=~/icml18_l1000

python f_train.py --train {pkl파일} --gene {gene embedding 파일의 위치} --vocab {} --save_dir {} --pre_vocab_dir {} --pre_model_dir {} --batch_size 16 --hidden_size 200

ex:)

CUDA_VISIBLE_DEVICES=1 nohup python f_train.py --train ../data/l1000/max40/processed --gene ../data/l1000/max40/train_emb40.txt --vocab ../data/l1000/max40/train_max40_vocab.txt --save_dir ./train_model/ --pre_vocab_dir ../data/l1000/max40/train_max40_vocab.txt --pre_model_dir ./pre_model/l1000_h200_b16_2epoch/model.iter-1 --batch_size 16 --hidden_size 200 > ./train_model/f_train_LOG.out &
```

- 
