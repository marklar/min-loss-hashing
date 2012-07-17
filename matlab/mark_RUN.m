
addpath utils;

%%%%%%%%%%%%%%%%%%%%%%%%%% Euclidean 22K LabelMe %%%%%%%%%%%%%%%%%%%%%%%%%%
data = mark_create_data('euc-22K-labelme', [], 0.01);

% in the paper n_trials = 10 used
n_trials = 3;
nBits = 64;

% number of learning iterations for validation
val_iter = 75;
% whether the hyper-plane offsets are zero during validation
% unused
% val_zerobias = 1;

for trialNum = 1:n_trials

  % ----- VALIDATION -----

  % -- Select RHO (heuristic) --
  % rho: hamming-distance threshold for neighborliness
  data2 = mark_create_training(data, 'train', 1);

  % 3rd arg: identity mtx.
  %   'precision' not used.
  %   'recall': used to select rho.
  [precision recall] = eval_LSH(nBits, data2);

  % recall: vector
  %   each cell corresponds to a different hamming distance
  %   its val: #good_recalled / #total_good
  % rho = # diff hamming dists w/ recall < 0.3.  WHY LT?
  rho = sum(recall < .3);	% bools: rho w/ <30% recall
  fprintf('automatic estimation of rho suggested rho = %d.\n', rho);
  clear data2;
  % done using data2.  only for determining rho.
  %-------------------------------

  % best_params: stores the most reasonable param settings.
  %   - initialize: start w/ same vals everytime
  best_params(trialNum).eta       = .1;    % learning rate
  best_params(trialNum).shrink_w  = 1e-4;  % weight decay
  best_params(trialNum).lambda    = .5;
  % changes: .rho, .shrink_w
  

  %-------------------------------------------------------
  
  % == VALIDATION for rho in hinge loss ==
  % rho might get large.
  % If want smaller hamming ball at retrieval time, set rho manually.
  % See alt. validation on lambda (instead of rho) for small databases.
  fprintf('validation for rho in hinge loss\n');

  % -- USE VALIDATION TO UPDATE RHO -- 
  % -- Create rho_set: options to try --
  % 1. select a STEP (of at least 1)
  step = round(nBits / 32);
  step(step < 1) = 1;
  % 2. create rho_set  (e.g.: [0 1 2 3 4])
  rho_set = rho + [-2 -1 0 +1 +2] * step;
  % 3. rm from rho_set those values < 1.
  rho_set(rho_set < 1) = [];
  % -- done creating rho_set --

  weightsTmp_rho = MLH(
    data,
    {rho_set, best_params(trialNum).lambda},
    nBits,
    'train',
    val_iter,
    0,  % num validations DURING training
    5,  % num validations AFTER  training
    best_params(trialNum).shrink_w,  % 0.0001
    0   % don't shrink learning rate
  );
  
  % -- if precision improves
  % set best_params(x).rho --
  best_ap = -1;
  for j = 1:numel(weightsTmp_rho)
    if (weightsTmp_rho(j).ap > best_ap)
      best_ap = weightsTmp_rho(j).ap;
      best_params(trialNum).rho = weightsTmp_rho(j).params.loss.rho;
    end
  end
  fprintf('Best rho (%d bits) = %d\n', nBits, best_params(trialNum).rho);
  % -- end: set rho -- 

  
  
  
  % -- USE VALIDATION TO UPDATE SHRINK_W --
  fprintf('validation for weight decay parameter\n');
  shrink_w_set = [.01 1e-3 1e-4 1e-5 1e-6];  % ever-decreasing
  weightsTmp_shrink_w = MLH(
    data,
    {best_params(trialNum).rho, best_params(trialNum).lambda},
    nBits,
    'train',
    val_iter,
    0,  % num validations DURING training
    5,  % num validations AFTER  training
    shrink_w_set,
    0   % don't shink eta (learning rate)
  );
  best_ap = -1;
  % try each shrink_w option in turn...
  for j = 1:numel(weightsTmp_shrink_w)
    % if precision improves, update shrink_w
    if (weightsTmp_shrink_w(j).ap > best_ap)
      best_ap = weightsTmp_shrink_w(j).ap;
      best_params(trialNum).shrink_w = weightsTmp_shrink_w(j).params.shrink_w;
    end
  end
  fprintf('Best weight decay (%d bits) = %.0d\n', nBits, ...
          best_params(trialNum).shrink_w);
  % -- end: set shrink_w --

  
  
  
  
  
  % ---- TRAINING on the train+val set -----
  % Here we use 500 learning iterations.
  % Using more might provide slightly better results.
  % In the paper, we used 2000.
  W(trialNum) = MLH(
    data,
    {best_params(trialNum).rho, best_params(trialNum).lambda},
    nBits,
    'trainval',
    500,  % max iterations
    5,  % num validations DURING training
    1,  % num validations AFTER  training
    [best_params(trialNum).shrink_w],
    1   % shrink learning rate?
  );
  
  % -- computing precision / recall --
  % save these for trialNum...
  % start w/ just 0s.
  precision(trialNum, :) = zeros(1, nBits+1);
  recall(   trialNum, :) = zeros(1, nBits+1);
  %
  [
      precision(trialNum, 1:nBits+1),
      recall(   trialNum, 1:nBits+1)
  ] =
      eval_linear_hash(W(trialNum).W, data);
  
  save res/mlh_euc-22K-labelme precision recall best_params W;
end
