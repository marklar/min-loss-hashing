function data = mark_create_training(data, trainset, doval)

% data:     modify and return
% trainset: 'train'
% doval:    true

Ntraining = data.Ntraining;
Xtraining = data.Xtraining;
Straining = data.Straining;

% 1/10th of training points -> use for validation
Ntest = min(ceil(Ntraining / 10), 5000);
Ntraining = Ntraining - Ntest;

% we re-define test set to be the validation set.
% this way all the evaluation code remains unchanged.

% _X_ test: only the final _X_ training cols
Xtest = Xtraining(:, Ntraining+1:end);

% _S_ test-training: from _S_ training
StestTraining = Straining(Ntraining+1:end, 1:Ntraining);
StestTraining(StestTraining  == -1) = 0;
Xtraining = Xtraining(:, 1:Ntraining);
Straining = Straining(1:Ntraining, 1:Ntraining);

Dtraining     = data.Dtraining(1:Ntraining, 1:Ntraining);  
DtestTraining = data.Dtraining(Ntraining+1:end, 1:Ntraining);

%-----

data.Xtraining     = Xtraining;
data.Straining     = Straining;
data.Ntraining     = Ntraining;
data.Dtraining     = Dtraining;
data.DtestTraining = DtestTraining;
data.Xtest         = Xtest;  
data.StestTraining = StestTraining;
data.Ntest         = Ntest;
