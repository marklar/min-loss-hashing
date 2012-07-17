function [score, recall] = evaluation2(S, Dhamm, maxn)

% Input:
%    S = true neighbors [Ntest * Ndataset], can be a full matrix NxN
%    Dhamm = estimated distances
%    maxn = number of distinct distance values to be considered
%
% Output:
%
%               exp. # of good pairs inside hamming ball of radius <= (n-1)
%  score(n) = --------------------------------------------------------------
%               exp. # of total pairs inside hamming ball of radius <= (n-1)
%
%               exp. # of good pairs inside hamming ball of radius <= (n-1)
%  recall(n) = --------------------------------------------------------------
%                          exp. # of total good pairs 

[Ntest, Ntrain] = size(S);
total_good_pairs = sum(S(:)==1);

[sD indsD] = sort(Dhamm(:));
[u ind] = unique(sD, 'first');
countu = [ind(2:end); numel(sD)+1] - ind;
hist = zeros(maxn+1, 1);
hist(u+1) = countu;
cumhist = [1; 1+cumsum(hist)];

% find pairs with similar codes
score = zeros(maxn+1,1);
retrieved_pairs = 0;
retrieved_good_pairs = 0;
for n = 1:(maxn+1)
    % exp. # of total pairs that have exactly the same code
    retrieved_pairs = retrieved_pairs + hist(n);
    
    %exp. # of good pairs that have exactly the same code
    retrieved_good_pairs = retrieved_good_pairs + sum(S(indsD(cumhist(n):cumhist(n+1)-1))==1);
    
    score(n) = retrieved_good_pairs/retrieved_pairs;
    recall(n)= retrieved_good_pairs/total_good_pairs;
end
