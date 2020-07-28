#!/bin/python
# coding: utf-8

# by Tegan Maharaj
#
# Uses/inspired by code from:
#    https://github.com/deeplearningathome/pytorch-language-model/blob/master/reader.py
#    https://github.com/ceshine/examples/blob/master/word_language_model/main.py
#    https://github.com/teganmaharaj/zoneout/blob/master/zoneout_word_ptb.py
#    https://github.com/teganmaharaj/IFT6135H19_assignment/blob/master/assignment2/ptb-lm.py
#    https://gist.github.com/spro/ef26915065225df65c1187562eca7ec4


import argparse
import time
import collections
import os
import sys
import torch
import torch.nn
from torch.autograd import Variable
import torch.nn as nn
import numpy
np = numpy

from data import load_data, data_iterator
from plotting import plot_curves

##############################################################################
#
# ARG PARSING AND EXPERIMENT SETUP
#
##############################################################################

parser = argparse.ArgumentParser(description='Interpolation and Extrapolation on 2- and 3- species food chains')

parser.add_argument('--train_data_path', type=str, default='esa_data/2sp_100ex_1var_TRAIN.npz',
                    help='location of train data')
parser.add_argument('--test_data_path', type=str, default='esa_data/2sp_100ex_1var_TEST.npz',
                    help='location of test data')
parser.add_argument('--optimizer', type=str, default='SGD_LR_SCHEDULE',
                    help='optimization algo to use; SGD, SGD_LR_SCHEDULE, ADAM')
parser.add_argument('--seq_len', type=int, default=1000,
                    help='number of timesteps over which BPTT is performed')
parser.add_argument('--batch_size', type=int, default=5,
                    help='size of one minibatch')
parser.add_argument('--initial_lr', type=float, default=2,
                    help='initial learning rate')
parser.add_argument('--hidden_size', type=int, default=200,
                    help='size of hidden layers. IMPORTANT: for the transformer\
                    this must be a multiple of 16.')
parser.add_argument('--save_best', action='store_true',
                    help='save the model for the best testidation performance')
parser.add_argument('--num_layers', type=int, default=2,
                    help='number of LSTM layers')
parser.add_argument('--emb_size', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--vocab_size', type=int, default=100,
                    help='size of vocabulary, or number of bins if discretizing data')
parser.add_argument('--grad_clip', type=float, default=0.25,
                    help='initial learning rate')
parser.add_argument('--num_epochs', type=int, default=40,
                    help='number of epochs to stop after')
parser.add_argument('--dp_keep_prob', type=float, default=0.35,
                    help='dropout *keep* probability (dp_keep_prob=0 means no dropout')
parser.add_argument('--zn_keep_prob', type=float, default=0.35,
                    help='zoneout *keep* probability (zn_keep_prob=0 means no zoneout')
parser.add_argument('--debug', action='store_true', 
                    help="Doesn't store experimental results, makes a small model.") 
parser.add_argument('--autoplot', type=str, default='lcs', 
                    help="Makes and saves a plot of learning curves") 
parser.add_argument('--save_dir', type=str, default='',
                    help='path to save the experimental config, logs, model \
                    This is automatically generated based on the command line \
                    arguments you pass and only needs to be set if you want a \
                    custom dir name')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')

args = parser.parse_args()
argsdict = args.__dict__
argsdict['code_file'] = sys.argv[0]

# Use the model, optimizer, and the flags passed to the script to make the 
# name for the experimental dir
print("\n########## Setting Up Experiment ######################")
#flags = [flag.lstrip('--') for flag in sys.argv[1:]]
args.train_data_path.split('/')[1].split('.')[0]
experiment_path = os.path.join(args.save_dir + '-'.join([
                     args.train_data_path.split('/')[1].split('.')[0],
                     args.test_data_path.split('/')[1].split('.')[0] ]))

# Increment a counter so that previous results with the same args will not
# be overwritten. Comment out the next four lines if you only want to keep
# the most recent results.
if not args.debug:
    i = 0
    while os.path.exists(experiment_path + "_" + str(i)):
        i += 1
    experiment_path = experiment_path + "_" + str(i)

# Creates an experimental directory and dumps all the args to a text file
os.mkdir(experiment_path)
print ("\nPutting log in %s"%experiment_path)
argsdict['save_dir'] = experiment_path
with open (os.path.join(experiment_path,'exp_config.txt'), 'w') as f:
    for key in sorted(argsdict):
        f.write(key+'    '+str(argsdict[key])+'\n')
#print_file(args.code_file, os.path.join(args.save_dir+'code_file.py'))

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

# Use the GPU if you have one
if torch.cuda.is_available():
    print("Using the GPU")
    device = torch.device("cuda") 
else:
    print("WARNING: You are about to run on cpu, and this will likely run out \
      of memory. \n You can try setting batch_size=1 to reduce memory usage")
    device = torch.device("cpu")


###############################################################################
#
# DATA LOADING & PROCESSING
#
###############################################################################

print('Training on '+ args.train_data_path + '\nTesting on '+ args.test_data_path)
train_data = load_data(args.train_data_path)
test_data = load_data(args.test_data_path)

if args.debug:
    print('Nevermind, actually debugging with 10 t/v simple pendulums')
    args.num_train=100
    args.num_test=100
    args.seq_len = 5000
    args.vocab_size=100
    from data import get_simple_pendulums_digitize
    train_data = get_simple_pendulums_digitize(args.num_train, args.seq_len, args.vocab_size)
    testid_data = get_simple_pendulums_digitize(args.num_test, args.seq_len, args.vocab_size)


###############################################################################
# 
# MODEL SETUP
#
###############################################################################
def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, Variable):
        return h.detach_()
    else:
        return tuple(repackage_hidden(v) for v in h)

class GRU(nn.Module):
  def __init__(self, input_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
    super(GRU, self).__init__()
    self.input_size = input_size
    self.output_size = input_size
    self.hidden_size = hidden_size
    self.seq_len = seq_len
    self.batch_size = batch_size
    self.vocab_size = vocab_size
    self.dp_keep_prob = dp_keep_prob
    self.num_layers = num_layers
    self.dropout = nn.Dropout(1 - dp_keep_prob)
    self.in_fc = nn.Linear(in_features=input_size,
                           out_features=hidden_size)
    self.rnn = nn.GRU(input_size=hidden_size,
                      hidden_size=hidden_size,
                      num_layers=num_layers,
                      dropout=1 - dp_keep_prob)
    self.sm_fc = nn.Linear(in_features=hidden_size,
                           out_features=self.output_size)
    self.init_weights()

  def init_weights(self):
    init_range = 0.1
    self.in_fc.bias.data.fill_(0.0)
    self.in_fc.weight.data.uniform_(-init_range, init_range)
    self.sm_fc.bias.data.fill_(0.0)
    self.sm_fc.weight.data.uniform_(-init_range, init_range)

  def init_hidden(self):
    weight = next(self.parameters()).data
    return weight.new(self.num_layers, self.batch_size, self.hidden_size).zero_()

  def forward(self, inputs, hidden):
    first = self.in_fc(inputs)
    rnn_out, hidden = self.rnn(first, hidden)
    rnn_out = self.dropout(rnn_out)
    logits = self.sm_fc(rnn_out.view(-1, self.hidden_size))
    return logits.view(self.seq_len, self.batch_size, self.output_size), hidden

input_size = train_data[0].shape[-1]
model = GRU(input_size=input_size, hidden_size=args.hidden_size, 
                seq_len=args.seq_len, batch_size=args.batch_size,
                vocab_size=args.vocab_size, num_layers=args.num_layers, 
                dp_keep_prob=args.dp_keep_prob)

model.to(device)

# LOSS FUNCTION
loss_fn = nn.MSELoss()
if args.optimizer == 'ADAM':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_lr)

# LEARNING RATE SCHEDULE    
lr = args.initial_lr
lr_decay_base = 1 / 1.15
m_flat_lr = 14.0 # we will not touch lr for the first m_flat_lr epochs


###############################################################################
# 
# DEFINE COMPUTATIONS FOR ONE EPOCH
#
###############################################################################

def run_epoch(model, data, is_train=False, lr=1.0):
    """
    One epoch of training/test (depending on flag is_train).
    """
    if is_train:
        model.train()
    else:
        model.eval()
    epoch_size = ((len(data[0]) // model.batch_size))# - 1) // model.seq_len
    start_time = time.time()
    hidden = model.init_hidden()
    hidden.to(device)
    costs = 0.0
    iters = 0
    losses = []

    # LOOP THROUGH MINIBATCHES
    #num_batches = 10 #num_ex // model.batch_size
    for step, (x, y) in enumerate(data_iterator(data, model.batch_size, model.seq_len)): #enumerate(ptb_iterator(data, model.batch_size, model.seq_len)):
        #import ipdb; ipdb.set_trace()
        inputs = torch.from_numpy(x.transpose((1,0,2))).float().contiguous().to(device)#.cuda()
        model.zero_grad()
        hidden = repackage_hidden(hidden)
        outputs, hidden = model(inputs, hidden)

        targets = torch.from_numpy(y.transpose((1,0,2))).float().contiguous().to(device)#.cuda()
        #tt = torch.squeeze(targets.view(-1, model.batch_size * model.seq_len))

        # LOSS COMPUTATION
        loss = loss_fn(outputs, targets) #outputs.contiguous().view(-1, model.vocab_size), tt)
        costs += loss.data.item() * model.seq_len
        losses.append(costs)
        iters += model.seq_len
        if args.debug:
            print(step, loss)
        if is_train:  # Only update parameters if training 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if args.optimizer == 'ADAM':
                optimizer.step()
            else: 
                for p in model.parameters():
                    if p.grad is not None:
                        p.data.add_(-lr, p.grad.data)
            if step % (epoch_size // 2) == 0:
                print('step: '+ str(step) + '\t' \
                    + 'loss: '+ str(costs) + '\t' \
                    + 'speed (wps):' + str(iters * model.batch_size / (time.time() - start_time)))
    return losses



###############################################################################
#
# RUN MAIN LOOP (TRAIN AND VAL)
#
###############################################################################

print("\n########## Running Main Loop ##########################")
train_losses = []
test_losses = []
best_test_so_far = np.inf
times = []

# In debug mode, only run one epoch
if args.debug:
    num_epochs = 1 
else:
    num_epochs = args.num_epochs

# MAIN LOOP
for epoch in range(2):#range(num_epochs):
    t0 = time.time()
    print('\nEPOCH '+str(epoch)+' ------------------')
    if args.optimizer == 'SGD_LR_SCHEDULE':
        lr_decay = lr_decay_base ** max(epoch - m_flat_lr, 0)
        lr = lr * lr_decay # decay lr if it is time

    # RUN MODEL ON TRAINING DATA
    train_loss = run_epoch(model, train_data, True, lr)

    # RUN MODEL ON VALIDATION DATA
    test_loss = run_epoch(model, test_data)

    # SAVE MODEL IF IT'S THE BEST SO FAR if we're doing that
    if sum(test_loss) < best_test_so_far:
        best_test_so_far = sum(test_loss)
        if args.save_best:
            param_path = os.path.join(args.save_dir, 'best_params.pt')
            print("Saving model parameters to "+param_path)
            torch.save(model.state_dict(), param_path)

    # LOG RESULTS
    train_losses.extend(train_loss)
    test_losses.extend(test_loss)
    #laimport ipdb; ipdb.set_trace()
    times.append(time.time() - t0)
    log_str = 'epoch: ' + str(epoch) + '\t' \
            + 'train loss: ' + str(sum(train_loss)) + '\t' \
            + 'test loss: ' + str(sum(test_loss))  + '\t' \
            + 'best test: ' + str(best_test_so_far) + '\t' \
            + 'time (s) spent in epoch: ' + str(times[-1])
    print(log_str)
    with open (os.path.join(args.save_dir, 'log.txt'), 'a') as f_:
        f_.write(log_str+ '\n')

# SAVE LEARNING CURVES
lc_path = os.path.join(args.save_dir, 'learning_curves.npy')
print('\nDONE\n\nSaving learning curves to '+lc_path)
np.save(lc_path, {'train_losses':train_losses,
                  'test_losses':test_losses})

if args.autoplot.startswith('lcs'):
    lcs_path = os.path.join(args.save_dir, 'lc_loss.png')
    losses_dict = {"Train": train_losses,
                   "Test": test_losses}
    labels_dict = {"y": "Loss (MSE)",
                   "x": "Epochs"}
    print("Saving learning curves to " + lcs_path)
    show = 'show' in args.autoplot
    plot_curves(losses_dict, labels_dict, lcs_path, showplot=show)
