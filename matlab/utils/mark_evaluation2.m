function [precision, recall] = mark_evaluation2(S, Dhamm, maxn)
%
% evaluation means...
%   given:
%     - the set of true neighbors
%     - hamming distances between their sketches
%
%

% Input:
%    S = true neighbors [Ntest * Ndataset], can be a full matrix NxN
%    Dhamm = estimated distances
%    maxn = number of distinct distance values to be considered
%
% Output:
%   for each distance value
%
%                  exp. # of good pairs inside hamming ball of radius <= (n-1)
%  precision(n) = --------------------------------------------------------------
%                  exp. # of total pairs inside hamming ball of radius <= (n-1)
%
%               exp. # of good pairs inside hamming ball of radius <= (n-1)
%  recall(n) = --------------------------------------------------------------
%                          exp. # of total good pairs 


% Dhamm: square mtx of pairwise similarity.
% sort every elem of Dhamm as a single list (col vector).
%   sD:    sorted distances
%   indsD: original indices of distances
[sD indsD] = sort(Dhamm(:));

% u:   unique values (distances)
% ind: idxs of first appearance of each uniq val
[u ind] = unique(sD, 'first');

% countUniq: num elems at each (distance) val.
countUniq = [ind(2:end); numel(sD)+1] - ind;

% hist: histogram of num instances for each val.
%   col vector.
%   maxn must be > |countUniq|
hist = zeros(maxn+1, 1);
% set a subset of vals in hist to the vals in countUniq
hist(u+1) = countUniq;

% cumhist: 'cumulative historgram'
%   cumulative sum:   same num elems as orig.
%     each val is sum of previous.
%   1 + cumsum(hist): add 1 to each elem.
cumhist = [1; 1+cumsum(hist)];

% create return val "precision".
% find pairs with similar codes
precision = zeros(maxn+1,1);
% why not also zero out "recall"?

% S: {0,1}.  Square mtx.  When pair is LABELED as match.
% t_g_p never changes.
total_good_pairs = sum(S(:)==1);

% updated with each loop iter...
retrieved_pairs = 0;
retrieved_good_pairs = 0;

% for each distance value...
for n = 1:(maxn+1)
  % exp. # of total pairs that are 'n' apart or less.
  retrieved_pairs = retrieved_pairs + hist(n);
  
  % foo: ORIG idxs of hamming distances.
  %      Distances, remember.  Not labeled as similar.
  foo = indsD(cumhist(n):cumhist(n+1)-1);
  % exp. # of good pairs that have exactly the same code
  retrieved_good_pairs = retrieved_good_pairs + sum( S(foo)==1 );
  
  precision(n) = retrieved_good_pairs/retrieved_pairs;
  recall(n)    = retrieved_good_pairs/total_good_pairs;
end
