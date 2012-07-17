function data = create_training(data, trainset, doval)

Ntraining = data.Ntraining;
Xtraining = data.Xtraining;
Straining = data.Straining;

if (strcmp(trainset, 'train'))
  % one tenth of the training points are used for validation
  Ntest = min(ceil(Ntraining / 10), 5000);
  Ntraining = Ntraining - Ntest;
  
  % we re-define test set to be the validation set.
  % this way all the evaluation codes remain unchanged.
  Xtest = Xtraining(:, Ntraining+1:end);
  StestTraining = Straining(Ntraining+1:end, 1:Ntraining);
  StestTraining(StestTraining  == -1) = 0;
  Xtraining = Xtraining(:, 1:Ntraining);
  Straining = Straining(1:Ntraining, 1:Ntraining);
  if (isfield(data, 'Dtraining'))
    Dtraining = data.Dtraining(1:Ntraining, 1:Ntraining);  
    DtestTraining = data.Dtraining(Ntraining+1:end, 1:Ntraining);
  end

  % if some kind of labeling exists e.g., class labels
  if (isfield(data, 'Ltraining'))
    Ltest = data.Ltraining(Ntraining+1:end, :);
    Ltraining = data.Ltraining(1:Ntraining, :);
  end
elseif (strcmp(trainset, 'trainval')) 
  % ordinary train and test sets are used
  if (isfield(data, 'Dtraining'))
    Dtraining = data.Dtraining;
  end
  if (isfield(data, 'DtestTraining'))
    DtestTraining = data.DtestTraining;
  end
  if (isfield(data, 'Xtest'))
    Ntest = data.Ntest;
    Xtest = data.Xtest;
    StestTraining = data.StestTraining;

    % if some kind of labeling exists e.g., class labels    
    if (isfield(data, 'Ltraining'))
      Ltest = data.Ltest;
      Ltraining = data.Ltraining;
    end
  end
else
  error('trainset should be either "trainval" or "train"');
end

data.Xtraining = Xtraining;
data.Straining = Straining;
data.Ntraining = Ntraining;
if (isfield(data, 'Dtraining'))
  data.Dtraining = Dtraining;
end
if (isfield(data, 'Ltraining'))
  data.Ltest = Ltest;
  data.Ltraining = Ltraining;
end
if (exist('Xtest') && doval)
  if (isfield(data, 'DtestTraining'))
    data.DtestTraining = DtestTraining;
  end
  data.Xtest = Xtest;  
  data.StestTraining = StestTraining;
  data.Ntest = Ntest;
end
if (~doval)
  if (isfield(data, 'Xtest'))
    data = rmfield(data, 'Xtest');
    data = rmfield(data, 'Ntest');
  end
  if (isfield(data, 'DtestTraining'))
    data = rmfield(data, 'DtestTraining');
  end
  if (isfield(data, 'StestTraining'))
    data = rmfield(data, 'StestTraining');
  end
end
