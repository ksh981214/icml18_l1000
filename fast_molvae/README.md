# Accelerated Training of Junction Tree VAE
Suppose the repository is downloaded at `$PREFIX/icml18-jtnn` directory. First set up environment variables:
```
export PYTHONPATH=$PREFIX/icml18-jtnn
```
The MOSES dataset is in `icml18-jtnn/data/moses` (copied from https://github.com/molecularsets/moses).

## Deriving Vocabulary
If you are running our code on a new dataset, you need to compute the vocabulary from your dataset.
To perform tree decomposition over a set of molecules, run
```
python ../fast_jtnn/mol_tree.py < ../data/moses/train.txt
```
This gives you the vocabulary of cluster labels over the dataset `train.txt`.

## Training
Step 1: Preprocess the data:
```
python preprocess.py --train ../data/moses/train.txt --split 100 --jobs 16
mkdir moses-processed
mv tensor* moses-processed
```
This script will preprocess the training data (subgraph enumeration & tree decomposition), and save results into a list of files. We suggest you to use small value for `--split` if you are working with smaller datasets.

Step 2: Train VAE model with KL annealing.
```
mkdir vae_model/
python vae_train.py --train moses-processed --vocab ../data/vocab.txt --save_dir vae_model/
```
Default Options:

`--beta 0` means to set KL regularization weight (beta) initially to be zero.

`--warmup 40000` means that beta will not increase within first 40000 training steps. It is recommended because using large KL regularization (large beta) in the beginning of training is harmful for model performance.

`--step_beta 0.002 --kl_anneal_iter 1000` means beta will increase by 0.002 every 1000 training steps (batch updates). You should observe that the KL will decrease as beta increases.

`--max_beta 1.0 ` sets the maximum value of beta to be 1.0.

`--save_dir vae_model`: the model will be saved in vae_model/

Please note that this is not necessarily the best annealing strategy. You are welcomed to adjust these parameters.

## vae_test.py

```
python vae_test.py --test ../data/l1000/mw500_pre_test/ --vocab ../data/l1000/mw500/vocab.txt --hidden_size 200 --trained_model ./pre_model/zinc1000_h200/model.iter-170000

if want Plot

python vae_test.py --test ../data/l1000/mw500_pre_test/ --vocab ../data/l1000/mw500/vocab.txt --hidden_size 200 --trained_model ./pre_model/zinc1000_h200/model.iter-170000 --plot 1

if want produce reconstructed molecules(very slow)

add '--make_generated 1'

```

## MJ_training

```
Need to change folder name with "processed" to "pos"

python mj_train.py --train ../data/l1000/max30/ --num_neg_folder 5 --gene ../data/l1000/max30/embedding_train_max30.txt --vocab ../data/l1000/max30/train_max30_vocab.txt --save_dir ./mj_model/ --pre_vocab_dir ../data/l1000/max30/train_max30_vocab.txt --pre_model_dir ./pre_model/minbyul/premodel_300_32_max30.iter-30000 --batch_size 32 --hidden_size 300
```
## mj_test.py

```
mode 0: molecule generation by Gene exp
mode 1: molecule reconstruction
mode -1(default): not recommend

python mj_test.py --test ../data/l1000/mw500_mj_test/ --vocab ../data/l1000/mw500/vocab.txt --hidden_size 200 --trained_model ./mj_model/zincl1000_mw500_h200/model.iter-20000 --batch_size 1 --mode 0 --sample_num 5
```

## Testing
To sample new molecules with trained models, simply run
```
python sample.py --nsample 30000 --vocab ../data/moses/vocab.txt --hidden 450 --model moses-h450z56/model.iter-700000 > mol_samples.txt
```
This script prints in each line the SMILES string of each molecule. `model.iter-700000` is a model trained with 700K steps with the default hyperparameters. This should give you the same samples as in [moses-h450z56/sample.txt](moses-h450z56/sample.txt). The result is as follows:
```
valid = 1.0
unique@1000 = 1.0
unique@10000 = 0.9992
FCD/Test = 0.42235413520261034
SNN/Test = 0.5560595345050097
Frag/Test = 0.996223352989786
Scaf/Test = 0.8924981494347503
FCD/TestSF = 0.9962165008703465
SNN/TestSF = 0.5272934146558245
Frag/TestSF = 0.9947901514732745
Scaf/TestSF = 0.10049873444911761
IntDiv = 0.8511712225340441
IntDiv2 = 0.8453088593783662
Filters = 0.9778
logP = 0.0054694810121243
SA = 0.015992957588069068
QED = 1.15692473423544e-05
NP = 0.021087573878091237
weight = 0.5403194879856983
```
