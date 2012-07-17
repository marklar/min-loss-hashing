% Copyright (c) 2011, Mohammad Norouzi and David Fleet

function [mean_ap final_W final_params] = mark_learnMLH(data, param, initW)

% Learn hash fns.
% Performs stochastic gradient descent to learn the hash params.
%
%
% Input:
%    data:  data for training the model.
%           split into two sets: training & validation.
%
%    param: params for how to perform.
%
%    initW: initial weight matrix -- random vals.  (same for each attempt.)
%
% Output:
%    mean_ap:      mean average precision over nval_after
%                  validation stages after training
%
%    final_W:      final weight matrix
%
%    final_params: params w/ additional components
%                  (mean_ap [redundantly], ave_error)
%    I WOULDN'T CALL THESE 'PARAMS'.  MORE LIKE 'STATS'.
%
%

nBits = param.nb;			% number of bits i.e, binary code length
initEta = param.eta;			% initial learning rate
shrink_eta = param.shrink_eta;		% whether shrink learning rate, as training proceeds
batchSize = param.batchSize;    	% mini-batch size
maxEpoch = param.maxiter;		% number of gradient update
                                        % epochs (each: 10k pairs)

zerobias = param.zerobias;		% whether offset terms are learned for hashing hyper-planes
                                        % or they all go through the origin
momentum = param.momentum;		% momentum term (between 0 and 1) for gradient update
shrink_w = param.shrink_w;		% weight decay parameter

% loss_func is structure. includes 'rho' & 'lambda'.
loss_func = param.loss;
lambda = loss_func.lambda;
rho = loss_func.rho;

Ntraining    = data.Ntraining;
NtrainingSqr = Ntraining^2;    % == number of training pairs???
Xtraining    = data.Xtraining;

% Find indices of all neighbor pairs.
% 'Pos' is for "positive".
% find(): returns linear indices corresponding to nonzero entries.
neighLinIdxs = find(data.Straining == 1);
nNeigh = numel(neighLinIdxs);

% Initial values of W, before training.
% Get initW -- either passed in, or use LSH.
if (exist('initW'))
  % initW is passed from ouside
  W = initW;
else
  % initialize W with LSH
  input_dim = size(Xtraining, 1);
  % randn: creates nBits x input_dim mtx.
  initW = [.1*randn(nBits, input_dim) zeros(nBits, 1)];
  % offset terms are initialized at zero
  % DON'T WE ALSO HAVE TO INITIALIZE 'W'?
end

% -- initialization --
% ntraining used only right here.
ntraining = 10^5; 	% 10k: tot num pairs per epoch
nCases = batchSize;
maxMiniBatches = floor(ntraining / nCases);  % num mini-batches
maxEpochs = maxEpoch + param.nval_after - 1;

% to be reported at end.
mean_ap = 0;
avg_err = 0;

% Winc: mtx w/ same dims as W
Winc = zeros(size(W));      % running ave of gradient
caseLinearIdxs = zeros(nCases,1);  % every instance in this batch

% gradient-update epochs
for epoch=1:maxEpochs
  
  % possibly update learning rate (eta)
  % (ever-increasing increments)
  if (shrink_eta)
    eta = initEta * (maxEpochs-epoch)/maxEpochs;
  else
    eta = initEta;
  end

  % MINI-BATCHES.
  % 'b' is unused.
  for b=1:maxMiniBatches
    % -- FOR THIS BATCH, SELECT INDICES OF SOME PAIRS --
    % Put them in caseLinearIdxs.
    % ???
    nCases2 = min(round(nCases * max(lambda - (nNeigh / NtrainingSqr), 0)), nCases); 
    % make the fraction of positive pairs at least lambda
    nCases1 = nCases - nCases2;
    % random selection of pairs
    caseLinearIdxs(1:nCases1) = ceil(rand(nCases1, 1)*NtrainingSqr);
    % selection of positive pairs
    % neighLinIdxs: linear indices of all neighbor pairs
    % What does this line do?
    caseLinearIdxs(nCases1+1:end) = neighLinIdxs(ceil(rand(nCases2, 1)*nNeigh));
    % -- done selecting caseLinearIdxs --
    
    % cases = ceil(rand(nCases, 1)*NtrainingSqr);
    % Ntraining = num rows & num cols

    % -- GET ACTUAL TRAINING INSTANCES --
    % ...from the pairs represented by caseLinearIdxs.
    % ind2sub(): linear indices -> mtx subscripts.
    % caseLinearIdx: for mtx (Ntraining x Ntraining).
    % Linear indices go DOWN COLUMNS.
    % 
    % x1nd: idxs of 1st instance in pair
    % x2nd: idxs of 2nd instance in pair
    %
    % Ntraining is a scalar.
    % [Ntraining Ntraining] is the SIZE of the mtx we're linearly
    % indexing into.
    % x1nd is just the array of row idxs.
    % x2nd is just the array of col idxs.
    [x1nd x2nd] = ind2sub([Ntraining Ntraining], caseLinearIdxs);

    % x1nd(:) == x1nd as a column vector.
    % From training, get instances (cols) of Xtraining.
    % x1: instances of 1st elem in pair
    % x2: instances of 2nd elem in pair
    x1 = Xtraining(:, x1nd(:));  % ... caseLinearIdxs' ROW indices.
    x2 = Xtraining(:, x2nd(:));  % ... caseLinearIdxs' COL indices.
    % -- end: get instances --

    % -- Get all S pairs corresponding to caseLinearIdxs. --
    % full(): (possibly) sparse mtx -> full mtx
    % Transpose: make column vector.
    % I assume that Straining was a sparse mtx, which means
    % an assoc: ((row,col), neighbor score).
    % neighOrNot: boolean ROW vector, showing neighbors-or-not
    % (just for this batch's cases).
    % it's a ROW because caseLinearIdxs is a COLUMN vec, so
    % data.Straining(caseLinearIdxs) is, too.  then take transpose.
    neighOrNot = full(data.Straining(caseLinearIdxs)');

    % -- Create y1, y2: 
    %
    % Why are these called y1, y2?  What is y?
    %
    % W:  mtx of hyperplanes.  (Cols: original features.)
    % x1: mtx of training examples (as cols).
    % Each elem in Wx1 is the inner (scalar) product of:
    %  * one hp w/
    %  * one training instance.
    % What does taking its sign tell you?
    % (What side you're on?)
    %
    % >> x1, x2 have same # of cols: nCases. <<
    % Add a 1 to the end (bottom) of each instance in x1.
    x1 = [x1; ones(1,nCases)];
    Wx1 = W*x1;  % W:   hyperplanes in rows.
                 % x1:  training instances in cols.
                 % Wx1: each elem is inner product of hp w/ instance.
    % y1: mtx (hp * training insts) ... indicating what?
    %     the sign of the inner product of:
    %       * each first instance (of a pair), w/
    %       * each hyperplane
    % y1 is zero where Wx1 is zero
    y1 = sign(Wx1);  % {-1,0,1}

    % Add a 1 to the end of each instance in x2.
    x2 = [x2; ones(1,nCases)];
    Wx2 = W*x2;
    % y2: mtx (hp * training insts) indicating which side of hp.
    % y2 is zero where Wx2 is zero
    y2 = sign(Wx2);
    % -- end: create y1, y2 --

    %
    % YES, ALL x1 VECTORS ARE NORMALIZED!
    % Hyperplanes are mean of 0, std dev of 1.  All values: * 0.1
    % Not "renormalized" anymore -- use weight decay.
    % (What does that mean?)
    %
    % If hps were normalized, then the dot product would be btwn -1.0 and 1.0.
    %
    
    % -- --
    %
    % This 'y' is diff from above.  Above: {-1,0,1}.  Here: R.
    % 
    % These are used to set the vals just below.
    % Dot products (hyperplane * instance), not mere signs.
    % Think of these as "regular" and "negated".
    y1plus  =  Wx1;			% y1 bits all on  ??
    y1minus = -Wx1;			% y1 bits all off ??
    y2plus  =  Wx2;			% y2 bits all on  ??
    y2minus = -Wx2;			% y2 bits all off ??

    % cat(dim, A, B):     make a 3-d tensor, with A "on top of" B.
    % max(tensor, [], 3): in 3rd dimension, returns largest elem
    % for each vector.  (In this case, it's largER, because there
    % are just two elements.)
    %
    % The values and indices of the largest elems.
    % So, best "score" when bits in y1 and y2 are the same.
    % Really, just the "abs" of their sum.
    %
    % SignsEqIdxs: {1,2}
    %   1 <- sum is positive
    %   2 <- sum is negative
    %
    % SignsEqIdxs: which sum-mtx it came from:
    %   1: pos+pos  -OR-
    %   2: neg+neg
    %
    [SignsEqSums SignsEqIdxs] = max(cat(3, y1plus+y2plus, y1minus+y2minus), ...
                          [], 3);
    % Best "score" when bits in y1 and y2 are different.
    % 
    % SignsNeqIdxs: {1,2}
    %   1 <- abs of 1st inner product is greater
    %   2 <- abs of 2nd inner product is greater
    %
    [SignsNeqSums SignsNeqIdxs] = max(cat(3, y1plus+y2minus, y1minus+y2plus), ...
                            [], 3);

    % SignsEqSums and SignsNeqSums are matrices of size nBits*nCases.
    % They're used only to create 'val'.  (What does 'val' mean?)
    %
    % sort(): sort each column vector.
    %   SortedDiffs:      the mtx with sorted columns.
    %   SortedDiffsIdxs:  orig idx of each val w/in the column.
    %
    % What is 'val' here?  What does the name mean?
    % again, sorted columns, and the orig indices w/in those columns.
    [SortedDiffs SortedDiffsIdxs] = sort(SignsNeqSums-SignsEqSums, 'descend');
    % -- --

    % -- CREATE LOSS MTX --
    %
    % A function of:
    %    - nBits
    %    - rho
    %    - neighOrNot ROW vec  <-  changes w/ each minibatch
    %        * size: 1 x nCases
    %
    % kron(): Kronecker tensor product.
    % K = kron(X,Y) returns the Kronecker tensor product of X & Y.
    % The result is a large array formed by taking all possible
    % products between the elements of X and those of Y.
    % If X is m-by-n and Y is p-by-q, then kron(X,Y) is m*p-by-n*q.
    %
    % neighOrNot == 1: ROW vec of bools (for where elem in neighOrNot is 1).
    %
    % creating the hinge-like loss for the positive and negative
    % pairs
    %
    % 'loss' is a huge mtx.
    % with loss for each pair.
    % by adding these two mtxs, we're just putting values in diff
    % places.

    % neighLoss:
    %   row vector of length nBits
    %   - first 'rho' vals: 0s.
    %   - remaining vals: fractional increments up to 1.0.
    %   the greater the # of bits different (m), the higher the val
    neighLoss = [zeros(1, rho) 1:(nBits-rho+1)] / (nBits-rho+1);  % neighLoss': 65 x 1
    positives = neighOrNot == 1;  % 1 x nCases (row vector)
    NeighborLoss = kron(positives, neighLoss');  % (nCases * 65) x 1

    % nonNeighLoss:
    %   row vector of length nBits
    %   - first rho+1 vals: descending ints (to 1)
    %     matlab:  (rho+1):-1:1
    %     haskell: [rho+1 .. 1]
    %   - remaining vals: 0s
    %   higher loss for too close.  no loss for non-neighbors.
    nonNeighLoss = [(rho+1):-1:1 zeros(1, nBits-rho)] / (rho+1);
    NonNeighborLoss = kron(neighOrNot == 0, nonNeighLoss');

    % size(Loss) == (nCases * 65) x 1
    Loss = NeighborLoss + NonNeighborLoss;

    %----------------------------------------
    % LOSS-ADJUSTED INFERENCE
    % max(): returns two row vectors:
    %    - val of max element of each col
    %    - idx of max elem of that col
    % cumsum(): in cols, each output elem is the cummulative sum of
    % previous.
    %
    % Find NUM BITS 
    % Cumsum mtx:
    %   Find cumulative sums of SortedDiffs mtx.
    %   SortedDiffs is already sorted w/in col in descending order.
    %   But then we're adding in the loss, so yes, we do need
    %       to take the max() of each column.
    %   Add a row of 0s to top of it.
    % Add cumsum & loss mtxs.
    % Find IDX of max elem of each col.
    %    nflip : vector; the IDXs of the max val in each col
    %    We subtract 1 because we've added a row of 0s to top.
    %    We needed to do that, in order to match it up with the
    %        'loss' mtx.
    fooMtx = [zeros(1, nCases); cumsum(SortedDiffs)];
    
    % foo: max value from each column.
    % nflip: the idx w/in column of that col's max value.
    [foo nflip] = max(fooMtx + Loss);
    nflip = nflip - 1;	% num diff bits in solution of loss-adjusted inference

    
    % (y1p, y2p) are the solutions to loss-adjusted inference ???
    y1p = zeros(size(y1));
    y2p = zeros(size(y2));    
    
    % tmp: matrix (nBits * nCases)
    %   - 1st repmat: (nBits * nCases): each col w/ values 1-64
    %   - 2nd repmat: 
    % 'TmpBools'
    tmp = repmat((1:nBits)',[1 nCases]) <= repmat(nflip, [nBits 1]);

    SortedDiffsIdxs = squeeze(SortedDiffsIdxs) + repmat(0:nBits:(nCases-1)*nBits,[nBits, 1]); % working out the indices
    
    % diffbits: col vector.  vals (which are, of course, indices) from
    % SortedDiffsIdxs.  go down columns, keeping vals (where tmp ==
    % true).
    % So diffbits is the indices of SortedDiffs
    diffbits = SortedDiffsIdxs(tmp);
    
    y1p(diffbits) = 2*(2-SignsNeqIdxs(diffbits))-1; % SignsNeqIdxs = 1/2 --> +1/-1
    y2p(diffbits) = 2*(SignsNeqIdxs(diffbits)-1)-1; % SignsNeqIdxs = 1/2 --> -1/+1
    samebits = find(y1p == 0);
    y1p(samebits) = 2*(2-SignsEqIdxs(samebits))-1;
    y2p(samebits) = 2*(2-SignsEqIdxs(samebits))-1;

    % nonzero_grad_1:
    %    one boolean per column.  (true if sum of column is non-zero.)
    %    ('sum' takes the sum OF EACH COLUMN.)
    nonzero_grad_1 = sum(abs(y1-y1p)) ~= 0;  % (~=): NOT EQUAL
    nonzero_grad_2 = sum(abs(y2-y2p)) ~= 0;

    % UNUSED?
    nonzero_grad = nonzero_grad_1 | nonzero_grad_2;

    % gradient
    % num-data-points * input-space (p+1)
    % must be same shape as W!
    % (select only some cols from x1, y1, etc.)
    %
    % y1: {-1,0,1}: sign(Wx1)
    % y2: {-1,0,1}: sign(Wx2)
    %
    % y1p, y2p: also matrices of same size (as W)
    % 
    grad = [x1(:,nonzero_grad_1) * (y1(:,nonzero_grad_1) - y1p(:,nonzero_grad_1))' + ...
	    x2(:,nonzero_grad_2) * (y2(:,nonzero_grad_2) - y2p(:,nonzero_grad_2))']';

    % ---- update W ----
    % zerobias -- offset terms: whether
    %  * learned for hashing hyper-planes, or
    %  * they all go thru the origin
    % vals in Winc start at 0, but will grow with each iter.
    Winc = (momentum * Winc) +
    % (eta * ...)  -- must be matrix of same shape as Winc (and W)
           (eta * (grad / nCases - shrink_w * [W(:,1:end-1) zeros(nBits, 1)]));
    if (zerobias)
      Winc(:,end) = 0;  % in final col, val = 0
    end
    W = W + Winc;
    
    % we don't re-normalize rows of W as mentioned in the paper anymore,
    % instead we use weight decay
    % i.e., L2 norm regularizer normW =
    % [repmat(sqrt(sum(W(:,1:end-1).^2,2)), [1 dtr+1])]; W = W./normW;
  end
  % -- end miniBatches --
  
  fprintf('(%3d/%.3f)', epoch, eta);
  
  if (param.nval_during && epoch < maxEpoch)
    if (mod(epoch, floor(maxEpoch/(param.nval_during))) == 0)
      % ignore return vals from eval().
      % why call it?  it doesn't seem to have any side effects.
      eval(epoch, eta, data, W, rho);
    end
  end
  
  if (param.nval_after && epoch >= maxEpoch)
    [ap err] = eval(epoch, eta, data, W, rho);
    avg_err = avg_err + err;
    mean_ap  = mean_ap  + ap;
  end
end

if (param.nval_after)
  mean_ap  = mean_ap  / param.nval_after;
  avg_err = avg_err / param.nval_after;
end

param.ap  = mean_ap;
param.err = avg_err;
final_W = W;
final_params = param;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ap err] = eval(epoch, eta, data, W, rho)

err = NaN; % err: return any type of error

[precision recall] = mark_eval_linear_hash(W, data);
precision(isnan(precision)) = 1;

ap = sum( [ ( precision(1:end-1) + precision(2:end)   ) / 2 ] .*
          [ ( recall(2:end)      - recall(1:end-1)    )     ]
        );
