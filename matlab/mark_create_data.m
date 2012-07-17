function data = mark_create_data(MODE, operand1, operand2, operand3)
data.MODE = MODE;

% Create the Euclidean 22K labelme dataset.
% (MAT-FILE?) FILENAME, VARIABLES FROM FILE TO LOAD
% what's 'nd'?
load('data/LabelMe_gist', 'ndxtrain', 'ndxtest', 'gist');

%-------- X --------
% In X, we want to store the data pts in columns.
% (They're not stored that way in gist.)
%
% There are:
%   22k data pts (cols),
%   each with 512 features (rows).
X = gist';
clear gist;

%
% ndxtrain:
%   vector of length 20019.  (missing 2000 of X)
%   ever-increasing indices, but not *every* int.
% ndxtest:
%   vector of length 2000.  the missing idxs from ndxtrain.
%   scrambled in order.  why?
%

% Xtraining & Xtest: store data pts in COLS.
% (select certain cols from X.)
% Disjoint subsets of X.
Xtraining = X(:, ndxtrain);
Xtest     = X(:, ndxtest);

% -- CENTER --
% Find the mean of each ROW.
gist_mean = mean(X, 2);
% Subtract each row's mean from all vals in row.
X = bsxfun(@minus, X, gist_mean);  % elem-by-elem minus.

% -- NORMALIZE --
% Normalize each column (data pt).
%
% Find length of each column.
%   Square all vals in X.
%   Sum each COLUMN's squares.  (The '1' says col.)
normX = sqrt(sum(X.^2, 1));
% Divide each val by its col's length.
X = bsxfun(@rdivide, X, normX);

%-------- Put "data" together --------

% operand1: []
% operand2: 0.01
% numel: num of elems in an array
data = construct_data(Xtraining, Xtest,
                      [numel(ndxtrain), numel(ndxtest)],
                      operand1, operand2, data);
% see bottom for construct_data(...)
data.gist_mean = gist_mean;

fprintf('done\n');

%--------------------------------------------
%--------------------------------------------

% construct the structure called 'data'
function data = construct_data(Xtraining, Xtest, sizeSets, avgNNeighbors, proportionNeighbors, data)

% avgNNeighbors: []  -- not set
%    ave num neighbors for each data pt
% proportionNeighbors: 0.01
%    fraction: (sim-pairs / total-pairs)
% --
% either avgNNeighbors or proportionNeighbors should be set.
% The other value should be empty ie., []
%
% avgNNeighbors: ave num of neighbors for each data pt
% proportionNeighbors: 0<=x<=1.  fraction of (similar pairs / total pairs)

% Just sizes of sets.
[Ntraining, Ntest] = deal(sizeSets(1), sizeSets(2));

% -- PAIRWISE DISTANCE MTX --
% Pairwise Euclidian distances between data pts
% (In Xtraining, each data pt is one COLUMN.)
% Strange thing is that we don't need an entire mtx;
% we need just less than 1/2 of it.
% So we're wasting computation to calculate distances,
% *and* we're possibly screwing up the calculations dependent
% upon this distance mtx.
Dtraining = distMat(Xtraining);

% -- CALC threshDist (using distance mtx) --
% using sim-scores, see how many (or what %) are w/in neighborhood
if (~isempty(avgNNeighbors))
  % use avgNNeighbors
  sortedD = sort(Dtraining, 2);   % for each pt, sort distances.
  threshDist = mean(sortedD(:,avgNNeighbors));  % ave of top X Eucl dist.
  data.avgNNeighbors = avgNNeighbors;
else
  % use proportionNeighbors
  sortedD = sort(Dtraining(:));   % ALL dists sorted in single col vec.
  threshDist = sortedD( ceil(proportionNeighbors * numel(sortedD)) );
  data.proportionNeighbors = proportionNeighbors;
end

% -- DO NOT UNDERSTAND WHAT THIS IS --
% distance mtx - between test and training
DtestTraining = distMat(Xtest, Xtraining); % size = [Ntest x Ntraining]

% elem-by-elem cmp -> logical array of same size.
% 'S': set of pairs for which sim labels exist.
data.Straining     = Dtraining     < threshDist;
data.StestTraining = DtestTraining < threshDist;

% store stuff in struct 'data'
data.Xtraining     = Xtraining;
data.Xtest         = Xtest;  
data.Ntraining     = Ntraining;
data.Ntest         = Ntest;
data.threshDist    = threshDist;
data.Dtraining     = Dtraining;
data.DtestTraining = DtestTraining;
