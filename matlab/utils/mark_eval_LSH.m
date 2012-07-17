% Random projection method by "Moses Charikar" which preserves cosine similarity
% reference: Similarity estimation techniques from rounding algorithms, M. Charikar, STOC 2002

function [precision recall] = mark_eval_LSH(nBits, data, S);

% S is the Gaussian covariance matrix
% By default S is the identity matrix

% create Hyperplanes
if ~exist('S')
  W = [
        % mtx (R ^ (q*p)) of normally distro'd randoms.
        randn(nBits, size(data.Xtraining, 1))
        % final col is all 0s
        zeros(nBits, 1)
      ];
else
  % multivariate normal random numbers.
  %
  % W: nBits-by-d mtx of random vecs
  %   chosen from the multivariate normal distribution with:
  %     * a common 1-by-d mean vector MU (of ZEROES), and
  %     * a common d-by-d covariance matrix SIGMA.
  %
  W = mvnrnd(
         zeros(1, size(data.Xtraining, 1)+1), % MU
         S,                                   % SIGMA
         nBits                                % cases
      );
end

[precision recall] = eval_linear_hash(W, data);
