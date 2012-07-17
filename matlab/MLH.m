% Copyright (c) 2011, Mohammad Norouzi and David Fleet

function [Wset] = MLH(data_in, loss_cell, nb, eta_set, momentum, size_batches_set, trainset, maxiter, ...
		      zerobias_set, nval_during, nval_after, verbose, shrink_w_set, shrink_eta)

% Performs validation on sets of parameters by calling appropriate instances of learnMLH function.
%
% Input:
%    data_in: data structure for training the model made by create_data.m
%    loss_cell: a cell array that determines the type of loss and its parameters. First element of
%      this cell defines loss type; either 'bre', or 'hinge'. For 'hinge' loss two other elements
%      are required defining 'rho' and 'lambda'. Using arrays for 'rho' and 'lamda' results in
%      validation on those parameters. For example, loss_cell being {'hinge', [3 4], [0 .5]} results
%      in validation on rho over 3 and 4, and on lambda over 0 and .5 (totally four configurations).
%    nb: number of bits
%    eta_set: choices for learning rate
%    momentum: momentum parameter for gradient descent (we always use .9)
%    size_batches_set: mini-batch size for gradient descent (we always use 100)
%    trainset: can be either 'train' or 'trainval'. Using 'train' splits the training set into train
%      and validation sets. Using 'trainval' performs training on the complete training set.
%    maxiter: number of iterations
%    zerobias_set: either 0 or 1, meaning whether the hashing hyper-planes' biases should be all
%      zero or should be learned. Both possibilities can be provided for validation.
%    nval_during: how many validations during training
%    nval_after: how many validation after training (to account for validation noise)
%    verbose: either 0 or 1, writing debug information or not
%
% Output:
%    A structure array storing sets of weight matrices (W), parameters (params), average precision
%      (ap), etc. learned by MLH


initW = [.1*randn(nb, size(data_in.Xtraining, 1)) zeros(nb, 1)]; % LSH
% same initialization is used for comparision of parameters
data = create_training(data_in, trainset, nval_during + nval_after);
if (verbose)
  display(data);
end

losstype = loss_cell{1};
if strcmp(losstype, 'hinge')
  rho_set = loss_cell{2};
  lambda_set = loss_cell{3};
  m = 1;
  for rho = rho_set
    for lambda = lambda_set
      loss_set(m).type = losstype;
      loss_set(m).rho = rho;
      loss_set(m).lambda = lambda;
      m = m+1;
    end
  end
else
  loss_set(1).type = losstype;
end

n = 1;
for size_batches = size_batches_set
for eta = eta_set
for shrink_w = shrink_w_set
for loss = loss_set
for zerobias = zerobias_set
  
  param.size_batches = size_batches;
  param.loss = loss;
  param.shrink_w = shrink_w;
  param.nb = nb;
  param.eta = eta;
  param.maxiter = maxiter;
  param.momentum = momentum;
  param.zerobias = zerobias;
  param.trainset = trainset;
  param.mode = data.MODE;
  param.Ntraining = data.Ntraining;
  param.nval_during = nval_during;
  param.nval_after = nval_after;
  param.shrink_eta = shrink_eta;
    
  [ap W Wall params] = learnMLH(data, param, verbose, initW);
  
  if (~verbose)
    if (numel(size_batches_set) > 1)
      fprintf('batch-size: %d  ', size_batches);
    end
    if (numel(loss_set) > 1)
      if strcmp(loss.type, 'hinge')
	fprintf('rho:%d / lambda:%.2f  ', loss.rho, loss.lambda);
      end
    end
    if (numel(eta_set) > 1)
      fprintf('eta: %.3f  ', eta);
    end
    if (numel(shrink_w_set) > 1)
      fprintf('shrink_w: %.0d  ', shrink_w);
    end
    fprintf(' --> ap:%.3f\n', ap);
  end
  
  Wset(n).ap = ap;
  Wset(n).W = W;
  Wset(n).params = params;
  Wset(n).mode = data_in.MODE;
  
  % Because PCA is not necessarily unique, we store the prinicipal components of the data with the
  % learned weights too.
  if (isfield(data_in, 'princComp'))
    Wset(n).princComp = data_in.princComp;
  end
  n = n+1;

end
end
end
end
end
