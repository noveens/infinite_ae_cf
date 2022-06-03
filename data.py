from scipy.sparse import csr_matrix
import jax.numpy as jnp
import numpy as np
import copy
import h5py
import gc

class Dataset:
    def __init__(self, hyper_params):
        self.data = load_raw_dataset(hyper_params['dataset'])
        self.set_of_active_users = list(set(self.data['train'][:, 0].tolist()))            
        self.hyper_params = self.update_hyper_params(hyper_params)

    def update_hyper_params(self, hyper_params):
        updated_params = copy.deepcopy(hyper_params)
        
        self.num_users, self.num_items = self.data['num_users'], self.data['num_items']
        self.num_interactions = self.data['num_interactions']

        # Update hyper-params to have some basic data stats
        updated_params.update({
            'num_users': self.num_users,
            'num_items': self.num_items,
            'num_interactions': self.num_interactions
        })

        return updated_params

    def sample_users(self, num_to_sample):
        if num_to_sample == -1: 
            ret = self.data['train_matrix']
        else: 
            sampled_users = np.random.choice(self.set_of_active_users, num_to_sample, replace=False)
            sampled_interactions = self.data['train'][np.in1d(self.data['train'][:, 0], sampled_users)]
            ret = csr_matrix(
                ( np.ones(sampled_interactions.shape[0]), (sampled_interactions[:, 0], sampled_interactions[:, 1]) ),
                shape = (self.num_users, self.num_items)
            )

        # This just removes the users which were not sampled
        return jnp.array(ret[ret.getnnz(1)>0].todense())

def load_raw_dataset(dataset, data_path = None, index_path = None):
    if data_path is None or index_path is None:
        data_path, index_path = [
            "data/{}/total_data.hdf5".format(dataset),
            "data/{}/index.npz".format(dataset)
        ]

    with h5py.File(data_path, 'r') as f: data = np.array(list(zip(f['user'][:], f['item'][:], f['rating'][:])))
    index = np.array(np.load(index_path)['data'], dtype = np.int32)

    def remap(data, index):
        ## Counting number of unique users/items before
        valid_users, valid_items = set(), set()
        for at, (u, i, r) in enumerate(data):
            if index[at] != -1:
                valid_users.add(u)
                valid_items.add(i)

        ## Map creation done!
        user_map = dict(zip(list(valid_users), list(range(len(valid_users)))))
        item_map = dict(zip(list(valid_items), list(range(len(valid_items)))))

        return user_map, item_map

    user_map, item_map = remap(data, index)

    new_data, new_index = [], []
    for at, (u, i, r) in enumerate(data):
        if index[at] == -1: continue
        new_data.append([ user_map[u], item_map[i], r ])
        new_index.append(index[at])
    data = np.array(new_data, dtype = np.int32)
    index = np.array(new_index, dtype = np.int32)

    def select(data, index, index_val):
        final = data[np.where(index == index_val)[0]]
        final[:, 2] = 1.0
        return final.astype(np.int32)

    ret = {
        'item_map': item_map,
        'train':  select(data, index, 0),
        'val': select(data, index, 1),
        'test': select(data, index, 2)
    }

    num_users = int(max(data[:, 0]) + 1)
    num_items = len(item_map)

    del data, index ; gc.collect()

    def make_user_history(arr):
        ret = [ set() for _ in range(num_users) ]
        for u, i, r in arr:
            if i >= num_items: continue
            ret[int(u)].add(int(i))
        return ret

    ret['train_positive_set'] = make_user_history(ret['train'])
    ret['val_positive_set'] = make_user_history(ret['val'])
    ret['test_positive_set'] = make_user_history(ret['test'])

    ret['train_matrix'] = csr_matrix(
        ( np.ones(ret['train'].shape[0]), (ret['train'][:, 0].astype(np.int32), ret['train'][:, 1].astype(np.int32)) ),
        shape = (num_users, num_items)
    )

    ret['val_matrix'] = csr_matrix(
        ( np.ones(ret['val'].shape[0]), (ret['val'][:, 0].astype(np.int32), ret['val'][:, 1].astype(np.int32)) ),
        shape = (num_users, num_items)
    )

    # Negatives will be used for AUC computation
    ret['negatives'] = [ set() for _ in range(num_users) ]
    for u in range(num_users):
        while len(ret['negatives'][u]) < 50:
            rand_item = np.random.randint(0, num_items)
            if rand_item in ret['train_positive_set'][u]: continue
            if rand_item in ret['test_positive_set'][u]: continue
            ret['negatives'][u].add(rand_item)
        ret['negatives'][u] = list(ret['negatives'][u])
    ret['negatives'] = np.array(ret['negatives'], dtype=np.int32)

    ret.update({
        'num_users': num_users,
        'num_items': num_items,
        'num_interactions': len(ret['train']),
    })

    print("# users:", num_users)
    print("# items:", num_items)
    print("# interactions:", len(ret['train']))

    return ret
