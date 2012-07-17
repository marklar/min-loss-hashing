function data = create_data(MODE, operand1, operand2, operand3)
data.MODE = MODE;

if (strcmp(MODE, 'uniform'))
  fprintf('Creating %dd %s dataset ... ', operand1, MODE);

  % Create a uniformly distributed synthetic dataset.
  % parameters
  dtr = operand1;
  Ntraining = 1000; % number training samples
  Ntest = 3000;     % number test samples
  averageNumberNeighbors = 50; % number of ground-truth neighbors for each training point (on average)
  aspectratio = ones(1, dtr);  % aspect ratio of different edge lenghts of the uniform hypercube

  Xtraining = rand([dtr, Ntraining]);
  for i=1:dtr
    Xtraining(i,:) = aspectratio(i) * Xtraining(i,:) - aspectratio(i)/2;
  end
  Xtest = rand([dtr, Ntest]);
  for i=1:dtr
    Xtest(i,:) = aspectratio(i)*Xtest(i,:) - aspectratio(i)/2;
  end

  data.aspectratio = aspectratio;
  data = construct_data(Xtraining, Xtest, [Ntraining, Ntest], averageNumberNeighbors, [], data); 
  % see bottom for construct_data(...)
  
elseif (strcmp(MODE, 'euc-22K-labelme'))
  fprintf('Creating %s dataset ... ', MODE);

  % Create the Euclidean 22K labelme dataset.
  load('data/LabelMe_gist', 'ndxtrain', 'ndxtest', 'gist');

  % data-points are stored in columns
  X = gist';
  clear gist;
  
  % center, then normalize data
  gist_mean = mean(X, 2);
  X = bsxfun(@minus, X, gist_mean);
  normX = sqrt(sum(X.^2, 1));
  X = bsxfun(@rdivide, X, normX);
  
  Xtraining = X(:, ndxtrain);
  Xtest = X(:, ndxtest);
  
  data = construct_data(Xtraining, Xtest, [numel(ndxtrain), numel(ndxtest)], operand1, operand2, data);
  % see bottom for construct_data(...)
  data.gist_mean = gist_mean;
  
elseif (strcmp(MODE, 'sem-22K-labelme'))
  fprintf('Creating %s dataset ... ', MODE);

  % Create the semantic 22K labelme dataset.
  load('data/LabelMe_gist', 'ndxtrain', 'ndxtest', 'DistLM', 'gist');
  
  % data-points are stored in columns
  X = gist';
  clear gist;

  % center, then normalize data
  gist_mean = mean(X, 2);
  X = bsxfun(@minus, X, gist_mean);
  normX = sqrt(sum(X.^2, 1));
  X = bsxfun(@rdivide, X, normX);
  if (~exist('operand3'))
    scale = 1;
  else
    scale = operand3;
  end
  X = X * scale;
  data.scale = scale;

  Xtraining = X(:, ndxtrain);
  Xtest     = X(:, ndxtest);
  Ntraining = numel(ndxtrain);
  Ntest = numel(ndxtest);

  Dtraining = -DistLM(ndxtrain, ndxtrain);
  DtestTraining = -DistLM(ndxtest, ndxtrain);

  sorted = sort(Dtraining, 2);
  D = sparse(bsxfun(@lt, Dtraining, sorted(:,operand1+1)));
  sorted2 = sort(DtestTraining, 2);
  D2 = sparse(bsxfun(@lt, DtestTraining, sorted2(:,operand1+1)));
  
  nNeighbors = operand1; % number of ground-truth neighbors for each training point (on average)

  data.MODE = MODE;
  data.Xtraining = Xtraining;
  data.Xtest = Xtest;
  data.Straining = D | D';
  data.StestTraining = D2;
  data.Ntraining = Ntraining;
  data.Ntest = Ntest;
  data.Dtraining = Dtraining;
  data.DtestTraining = DtestTraining;
  data.averageNumberNeighbors = nNeighbors;
  data.max_care = operand2; % used for cross-validation in eval_labelme function

elseif (strcmp(MODE, 'kulis'))
  fprintf('Creating %s-%s dataset ... ', MODE, operand1);
  
  % From Brian Kulis's code; Preparing datasets from the BRE paper
  data.MODE = [MODE, ' - ', operand1];
  X = load(['data/kulis/', operand1, '.mtx'])';

  averageNumberNeighbors = 50; % number of ground-truth neighbors for each training point (on average)
  Ntraining = 1000;
  Ntest = min(3000, size(X,2)-1000);

  % center, then normalize data
  X = bsxfun(@minus, X, mean(X,2));
  normX = sqrt(sum(X.^2, 1));
  X = bsxfun(@rdivide, X, normX);
  
  % each time a new permuatation of data is used
  rp = randperm(size(X,2));
  trIdx = rp(1:Ntraining);
  testIdx = rp(Ntraining+1:Ntraining+Ntest);
  Xtraining = X(:, trIdx);
  Xtest = X(:, testIdx);
  
  data = construct_data(Xtraining, Xtest, [Ntraining, Ntest], averageNumberNeighbors, [], data);
  % see bottom for construct_data(...)
  
else
  error('The given mode is not recognized.\n');
end

fprintf('done\n');


function data = construct_data(Xtraining, Xtest, sizeSets, avgNNeighbors, proportionNeighbors, data)

% either avgNNeighbors or proportionNeighbors should be set. The other value should be empty ie., []
% avgNNeighbors is a number which determines the average number of neighbors for each data point
% proportionNeighbors is between 0 and 1 which determines the fraction of [similar pairs / total pairs]

[Ntraining, Ntest] = deal(sizeSets(1), sizeSets(2));
Dtraining = distMat(Xtraining);

if (~isempty(avgNNeighbors))
  sortedD = sort(Dtraining, 2);
  threshDist = mean(sortedD(:,avgNNeighbors));
  data.avgNNeighbors = avgNNeighbors;
else
  sortedD = sort(Dtraining(:));
  threshDist = sortedD(ceil(proportionNeighbors * numel(sortedD)));
  data.proportionNeighbors = proportionNeighbors;
end

DtestTraining = distMat(Xtest, Xtraining); % size = [Ntest x Ntraining]

data.Xtraining = Xtraining;
data.Xtest = Xtest;  
data.Straining = Dtraining < threshDist;
data.StestTraining = DtestTraining < threshDist;

data.Ntraining = Ntraining;
data.Ntest = Ntest;
data.threshDist = threshDist;
data.Dtraining = Dtraining;
data.DtestTraining = DtestTraining;
