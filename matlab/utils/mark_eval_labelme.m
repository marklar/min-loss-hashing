function [ap p_code] = mark_eval_labelme(W, data);

% _X_
%   B1,B2: *logical* mtxs.  {0,1}.
%   'B' stands for "bit"?
%
%   ones: mtx w/ size (1*x)
%   ';' -- used inside brackets to end rows
B1 = (W * [data.Xtraining; ones(1, data.Ntraining)]) > 0;
B2 = (W * [data.Xtest;     ones(1, data.Ntest)    ]) > 0;

% _N_
% split Ntraining into:
%   - ndxtrain
%   - ndxtest
ndxtrain = 1 : data.Ntraining;
ndxtest  = data.Ntraining+1 : data.Ntraining+data.Ntest;

% create 'code'.
code(:,ndxtrain) = B1;  % fill first ndxtrain cols w/ B1
code(:,ndxtest)  = B2;  % fill next  ndxtest  cols w/ B2
code = compactbit(code);
code = uint8(code);   % convert every val to uint8

P_code = zeros(numel(ndxtrain), numel(ndxtest));
% compute Hamming distance values
D_code = hammingDist(code(:,ndxtest),code(:,ndxtrain));

for n = 1:length(ndxtest)
  ndx = ndxtest(n);
  
  [foo, j_code] = sort(D_code(n, :), 'ascend'); % I assume that smaller distance means closer
  j_code = ndxtrain(j_code);
  
  % get groundtruth sorting
  D_truth = data.DtestTraining(ndx-data.Ntraining,:);
  [foo, j_truth] = sort(D_truth);
  j_truth = ndxtrain(j_truth);
  
  % evaluation
  P_code(:,n) = neighbors_recall(j_truth, j_code);
end

p_code = mean(P_code,2);
ap = mean(p_code(1:data.max_care));
