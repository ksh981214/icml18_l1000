# icml18_l1000

- 학습을 돌리기위한 최소의 data/file을 남겨놓음.

```
export PYTHONPATH=~/icml18_l1000

python f_train.py --train {pkl파일} --gene {embedding.txt의 위치} --vocab {} --save_dir {} --pre_vocab_dir {} --pre_model_dir {} --batch_size 16 --hidden_size 200

python f_train.py --train ../data/l1000_121/train_gene_processed --gene ../data//l1000_121 --vocab ../data/l1000_121/train_vocab_121.txt --save_dir ./train_model/ --pre_vocab_dir ../data/l1000_121/train_vocab_121.txt --pre_model_dir ./pre_model/l1000_h200/model.iter-25000 --batch_size 16 --hidden_size 200
```

- 
