% Random projection method by "Moses Charikar" which preserves cosine similarity
% reference: Similarity estimation techniques from rounding algorithms, M. Charikar, STOC 2002

function [p0 r0] = eval_LSH(nbits, data, S);

% S is the Gaussian covariance matrix
% By default S is the identity matrix

if ~exist('S')
  W = [randn(nbits, size(data.Xtraining, 1)) zeros(nbits, 1)];
else
  W = mvnrnd(zeros(1, size(data.Xtraining, 1)+1), S, nbits);
end

[p0 r0] = eval_linear_hash(W, data);
