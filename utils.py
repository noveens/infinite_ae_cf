import numpy as np
from collections import defaultdict

def get_common_path(hyper_params):
    ret = "{}_users_{}_depth_{}_".format(
        hyper_params['dataset'], hyper_params['user_support'],
        hyper_params['depth']
    )
    
    if hyper_params['grid_search_lamda']: ret += "grid_search_lamda_"
    else: ret += "lamda_{}_".format(hyper_params['lamda'])
    
    ret += "seed_{}".format(hyper_params['seed'])
    return ret

def get_item_count_map(data):
    item_count = defaultdict(int)
    for u, i, r in data.data['train']: item_count[i] += 1
    return item_count

def get_item_propensity(hyper_params, data, A = 0.55, B = 1.5):
    item_freq_map = get_item_count_map(data)
    item_freq = [ item_freq_map[i] for i in range(hyper_params['num_items']) ]
    num_instances = hyper_params['num_interactions']

    C = (np.log(num_instances)-1)*np.power(B+1, A)
    wts = 1.0 + C*np.power(np.array(item_freq)+B, -A)
    return np.ravel(wts)

def file_write(log_file, s, dont_print=False):
    if dont_print == False: print(s)
    if log_file is None: return
    f = open(log_file, 'a')
    f.write(s+'\n')
    f.close()

def log_end_epoch(hyper_params, metrics, step, time_elpased, metrics_on = '(TEST)', dont_print = False):
    string2 = ""
    for m in metrics: string2 += " | " + m + ' = ' + str("{:2.4f}".format(metrics[m]))
    string2 += ' ' + metrics_on

    ss  = '| end of step {:4d} | time = {:5.2f}'.format(step, time_elpased)
    ss += string2
    file_write(hyper_params['log_file'], ss, dont_print = dont_print)
