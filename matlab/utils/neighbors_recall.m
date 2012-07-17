function P = neighbors_recall(trueRank, rank)
%
% compute P(i in Saprox(M) | i in S(N)) as a function of M
% we fix M = 50;
%
% Input:
%   - trueRank: indices from closest neighbors to far
%   - rank: output ranking of the algorithm
% Output:
%   - P: vector of dimension [600 1] = P(i in Saprox(M) | i in S(N)), for N=1:600

[nimagestest, sizedatabase] = size(trueRank);

M = 50;
N = M;

j = trueRank(:,1:N);
jj = zeros(size(rank));
for n = 1:nimagestest
  jj(n,:) = ismember(rank(n,:), j(n,:));
end
Pi = single(cumsum(jj,2)/N);

P = mean(Pi,1);
