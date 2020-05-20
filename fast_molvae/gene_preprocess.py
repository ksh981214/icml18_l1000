import pickle
import argparse

def gene_preprocess(save_path,file_name, num_splits):
    with open(save_path+file_name) as f:
        data = [list(map(float, emb.split())) for emb in f] #LIST[LIST]
    le = (len(data) + num_splits - 1) / num_splits

    for split_id in xrange(num_splits):
        st = split_id * le
        sub_data = data[st : st + le]

        with open(save_path + 'gene/gene-%d.pkl' % split_id, 'wb') as f:
            pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser()
    '''
        --save_path : ../data/l1000/max40/
        --file_name : embedding_train_max40.txt
        --num_splits
    '''
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--file_name', required=True)
    parser.add_argument('--num_splits', type=int, default=100)

    args = parser.parse_args()
    print args

    save_path = args.save_path
    file_name = args.file_name
    num_splits = args.num_splits

    #save txt
    gene_preprocess(save_path, file_name, num_splits)

if __name__ == "__main__":
    main()
