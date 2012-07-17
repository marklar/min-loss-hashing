function Dh = mark_hammingDist(B1, B2)
%
% Compute hamming distance between two sets of binary codes (B1, B2)
%
% Dh = hammingDist(B1, B2);
%
% Input:
%    B1, B2: compact bit vectors.
%        Each binary code is 1 col comprised of uint8 numbers.
%    size(B1) = [nwords, ndatapoints1]
%    size(B2) = [nwords, ndatapoints2]
%    It is faster if ndatapoints1 < ndatapoints2
% 
% Output:
%    Dh = hamming distance. 
%    size(Dh) = [ndatapoints1, ndatapoints2]

% example query:
%    Dhamm = hammingDist(B2, B1);
%    size(Dhamm) = [Ntest x Ntraining]

% look-up table:
bit_in_char = uint16([...
    0 1 1 2 1 2 2 3 1 2 2 3 2 3 3 4 1 2 2 3 2 3 ...
    3 4 2 3 3 4 3 4 4 5 1 2 2 3 2 3 3 4 2 3 3 4 ...
    3 4 4 5 2 3 3 4 3 4 4 5 3 4 4 5 4 5 5 6 1 2 ...
    2 3 2 3 3 4 2 3 3 4 3 4 4 5 2 3 3 4 3 4 4 5 ...
    3 4 4 5 4 5 5 6 2 3 3 4 3 4 4 5 3 4 4 5 4 5 ...
    5 6 3 4 4 5 4 5 5 6 4 5 5 6 5 6 6 7 1 2 2 3 ...
    2 3 3 4 2 3 3 4 3 4 4 5 2 3 3 4 3 4 4 5 3 4 ...
    4 5 4 5 5 6 2 3 3 4 3 4 4 5 3 4 4 5 4 5 5 6 ...
    3 4 4 5 4 5 5 6 4 5 5 6 5 6 6 7 2 3 3 4 3 4 ...
    4 5 3 4 4 5 4 5 5 6 3 4 4 5 4 5 5 6 4 5 5 6 ...
    5 6 6 7 3 4 4 5 4 5 5 6 4 5 5 6 5 6 6 7 4 5 ...
    5 6 5 6 6 7 5 6 6 7 6 7 7 8]);

% -- B1, B2: *compact* hashes --
% n1:     num cols in B1
% n2:     num cols in B2
% nwords: num rows in B2  (should be same for both B1,B2)
n1 = size(B1, 2);
[nwords n2] = size(B2);

% Dh: the distances.
%   n1-by-n2 mtx, initially of 0s.
%   to be updated in nested 'for' loops below.
Dh = zeros([n1 n2], 'uint16');

% for each col in B2
for j = 1:n2
  % for each row
  for n = 1:nwords
    % y+1 could become 256, which causes an error for uint8 y
    
    % bitxor: 0 if same, 1 if different
    %   args: (1) row of B1, (2) single elem of B2.
    y = uint16( bitxor( B1(n,:), B2(n,j))' );

    % update Dh col j.
    Dh(:,j) = Dh(:,j) + bit_in_char(y+1)';
  end
end
