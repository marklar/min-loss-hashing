function data = do_pca(data, nb);

% performing PCA on the data structure
npca = min(nb, size(data.Xtraining, 1));
opts.disp = 0;
[pc, l] = eigs(cov(data.Xtraining'), npca, 'LM', opts);
data.Xtraining = pc' * data.Xtraining;
if (isfield(data, 'Xtest'))
  data.Xtest = pc' * data.Xtest;
end

data.princComp = pc;
