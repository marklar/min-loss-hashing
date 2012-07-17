function [precision recall] = mark_eval_linear_hash(W, data)

% W: the params to the hash fns.

% pull stuff out of 'data'
% *number* of test/training points
Ntest         = data.Ntest;
Ntraining     = data.Ntraining;
% actual test/training points
Xtest         = data.Xtest;
Xtraining     = data.Xtraining;
% labeled data
%   0: not neighbors
%   1: neighbors
StestTraining = data.StestTraining;

% -- COMPUTE LINEAR HASHES --
% binary quantization
%   add extra row of 1s because ...?
B1 = (W * [Xtraining; ones(1,Ntraining)]) > 0;
B2 = (W * [Xtest;     ones(1,Ntest)    ]) > 0;

% -- AND COMPACT THEM --
% compact: byte-size {0,1} -> bits.
B1 = compactbit(B1);
B2 = compactbit(B2);

% -- CALC PAIRWISE DIST --
% find pairwise distances: tween test & training.
% why reverse the order?!?
Dhamm = hammingDist(B2, B1);

% calc precision and recall.
%   - StestTraining: true neighbors
%   - Dhamm:         estimated distances
%   - num rows:      number of distinct distance vals to be considered.
[precision recall] = evaluation2(StestTraining, Dhamm, size(W,1));
precision = full(precision');   % transpose!
recall    = full(recall);
