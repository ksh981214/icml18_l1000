# icml18_l1000

- 학습을 돌리기위한 최소의 data/file을 남겨놓음.

```
export PYTHONPATH=~/icml18-jtnn

python mj_train.py --train ../data/l1000/max30/ --num_neg_folder 5 --vocab ../data/l1000/max30/train_max30_vocab.txt --save_dir ./mjmodel/ --pre_vocab_dir ../data/l1000/max30/train_max30_vocab.txt --pre_model_dir ./premodel/minbyul/premodel_200_32_max30.iter-25000 --batch_size 32 --hidden_size 200

- 
