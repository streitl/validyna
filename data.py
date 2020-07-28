import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os
import collections

import matplotlib.collections as mcoll
import matplotlib.path as mpath


###############################################################################
#
# MAKE DATASETS 
#
###############################################################################

def make_datasets(seq_len=1000, 
                  len_transitory_discard=4000, 
                  num_speciess=[2,3], 
                  initial_initial_state=[0.1,0.5,0.5],
                  variance_orders=[0.01,0.1,1,10], # sample between initial state and initial_state + this number from the initial-initial conditions
                  num_exs=[10,100]#,1000,5000] # number of sequences (examples) to sample 
                        ):

    # 2-species ecological network
    def LotkaVolterra(state,t):
        x = state[0]  # species 1
        y = state[1]  # species 2
        alpha = 0.1
        beta  = 0.1
        sigma = 0.1
        gamma = 0.1

        # dynamics
        xd = x*(alpha - beta*y)
        yd = -y*(gamma - sigma*x)
        return [xd,yd]

    # 3-species ecological network
    def HastingsPowell(state,t):
        x = state[0]  # species 1
        y = state[1]  # species 2
        z = state[2]  # species 3
        a1 = 5.
        b1 = 3. # varies from 2.0 to 6.2
        a2 = 0.1
        b2 = 2.
        d1 = 0.4
        d2 = 0.01

        # dynamics
        xd = x*(1-x) - a1*x*y /(1 + b1*x)
        yd = a1*x*y / (1+ b1*x) - \
             a2*y*z / (1+ b2*y) - \
             d1*y
        zd = a2*y*z / (1+ b2*y) - \
             d2*z

        return [xd, yd, zd]    

    # Time range (including portion of transitory dynamics to discard and
    # adding 1 so we can offset the targets for next-step prediction)
    t = np.arange(0, len_transitory_discard + seq_len + 1, 1)

    # For both ecosystem network models ...
    for num_species in num_speciess:
        if num_species == 2:
            ode = LotkaVolterra
            initial_state = np.asarray(initial_initial_state[:2])
        else:
            ode = HastingsPowell
            initial_state = np.asarray(initial_initial_state)
    
        # for different dataset sizes ...
        for num_ex in num_exs:  # Train and Test are the same size
    
            # for different variance orders ...
            for variance in variance_orders:
                
                # ... Sample initial states
                initial_states = np.random.uniform(initial_state,
                                                   initial_state + variance,
                                                   (num_ex*2, num_species))  # *2 for both Train & Test
                # solve the ODE to get Training set
                train_states = []
                for state0 in initial_states[:num_ex]:
                    state = odeint(ode, state0, t)
                    train_states.append(state[len_transitory_discard:])
                to_save = np.asarray(train_states)
                filename = 'data/' + \
                           '_'.join([str(num_species)+'sp',
                                     str(num_ex)+'ex',
                                     str(variance)+'var',
                                     'TRAIN'])
                np.savez(filename, state=to_save)

                # and Test set
                test_states = []
                for state0 in initial_states[num_ex:]:
                    state = odeint(ode, state0, t)
                    test_states.append(state[len_transitory_discard:])
                to_save = np.asarray(test_states)
                filename = 'data/' + \
                           '_'.join([str(num_species)+'sp',
                                     str(num_ex)+'ex',
                                     str(variance)+'var',
                                     'TEST'])
                np.savez(filename, state=to_save)


def load_data(dataset_path):
    _d = np.load(dataset_path)['state']  # n,t,c (num_ex, time, state)
    inputs = _d.reshape(_d.shape[0], _d.shape[2], _d.shape[1])[:,:,:-1].reshape(
                        _d.shape[0], _d.shape[1]-1, _d.shape[2])
    targets = _d.reshape(_d.shape[0], _d.shape[2], _d.shape[1])[:,:,1:].reshape(
                        _d.shape[0], _d.shape[1]-1, _d.shape[2])
    return inputs, targets 


def load_data_digitize(dataset, vocab_size):

    # Puts floats in indexed discrete bins so that word embeddings can be used
    def _digitize(data, vocab_size=100):
        # normalize to range (0,1)
        data = (data - np.min(data)) / (np.max(data) - np.min(data)) 
        the_bins = np.histogram_bin_edges(data, bins=vocab_size, range=(0,1.))
        return np.digitize(data, the_bins)

    _d = _digitize(np.load(dataset)['state'], vocab_size) # n,t,c (num_ex, time, state)
    inputs = _d.reshape(_d.shape[0], _d.shape[2], _d.shape[1])[:,:,:-1].reshape(
                        _d.shape[0], _d.shape[1]-1, _d.shape[2])
    targets = _d.reshape(_d.shape[0], _d.shape[2], _d.shape[1])[:,:,1:].reshape(
                        _d.shape[0], _d.shape[1]-1, _d.shape[2])
    return inputs, targets 


def data_iterator(data, batch_size, seq_len):
    #import ipdb; ipdb.sets_trace()
    num_ex = len(data[0])
    num_batches = num_ex // batch_size
    for i in range(num_batches):
        x,y = data
        a = x[batch_size * i:batch_size * (i + 1)]
        b = y[batch_size * i:batch_size * (i + 1)]
        yield (a,b)