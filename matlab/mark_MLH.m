% Copyright (c) 2011, Mohammad Norouzi and David Fleet

function [weightsSet] = mark_MLH(data_in, loss_cell, nBits, trainset,
                                 maxiter, ...
                                 nval_during, nval_after, 
                                 shrink_w_set, shrink_eta_p)

% Select 10% of training set as a validation set.
% Choose: epsilon and loss hyperparams: rho & lambda
%   by validation on a few candidate choices.
%   epsilon: boost loss, to balance w/ scoring fn.
% We allow rho to increase linearly with the code length.
%
% Epochs:
%   each includes a random sample of 100k pt pairs.
%   validation: 100 epochs
%   training:  2000 epochs
%

% Performs validation on sets of parameters by calling appropriate
% instances of learnMLH function.
%
% Input:
%    data_in: data structure for training the model made by
%    create_data.m
%
%    loss_cell: 2 vals:
%       * its parameters: 'rho', 'lambda'
%      rho and lambda can be arrays -> results in validation on
%      those parameters.
%      e.g., {[3 4], [0 .5]} -> validation:
%        on rho over 3 and 4, and
%        on lambda over 0 and .5
%      for a total of 4 configs.
%
%    nBits: number of bits
%
%    trainset: can be either 'train' or 'trainval'.  (names backwards?)
%      'train':    splits training set into train & validation sets
%      'trainval': performs training on the complete training set
%
%    maxiter: number of iterations  ("patience"?)
%      what's an iteration?
%
%    nval_during: how many validations during training
%
%    nval_after: how many validations after training (to account
%    for validation noise)
%
%
% Output:
%    A structure array storing sets of weight matrices (W), parameters (params), average precision
%      (ap), etc. learned by MLH

batchSize = 100;  % for gradient descent
momentum  = 0.9;  % for GD.  HIGH!  (mostly insensitive to local gradient)
eta       = 0.1;  % training rate

% initW: normally distributed random numbers.
%   #rows = nBits
%   #cols = Xtraining #rows
%   then add 1 more col (of 0s)
initW = [.1*randn(nBits, size(data_in.Xtraining, 1)) zeros(nBits, 1)];
% -- LSH same initialization is used for comparision of params

data = create_training(data_in, trainset, nval_during + nval_after);

% Try all these diff combinations.
n = 1;
for shrink_w = shrink_w_set

  % INPUTS to learning
  % 'param' (singular) is the input
  param.batchSize = batchSize;
  param.loss = loss;
  param.shrink_w = shrink_w;
  param.nBits = nBits;
  param.eta = eta;
  param.maxiter = maxiter;
  param.momentum = momentum;
  param.zerobias = 1;
  param.trainset = trainset;
  param.mode = data.MODE;
  param.Ntraining = data.Ntraining;
  param.nval_during = nval_during;
  param.nval_after = nval_after;
  param.shrink_eta = shrink_eta_p;
  
  % LEARN
  % 'params' (plural) is the OUTPUT
  [avePrecision W params] = mark_learnMLH(data, param, initW);
  
  % OUTPUTS - put in weightsSet
  weightsSet(n).ap     = avePrecision;  % mean ave. precision (over validation stages)
  weightsSet(n).W      = W;             % weights
  weightsSet(n).params = params;        
  weightsSet(n).mode   = data_in.MODE;  % 'training'/'test', etc.
  
  n = n+1;
end
