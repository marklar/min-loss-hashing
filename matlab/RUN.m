% doEuc22K: whether to perform experiments on Euclidean 22K LabelMe
% doSem22K: whether to perform experiments on Semantic 22K LabelMe
% doSmallDB: whether to perform experiments on 6 small DBs
doEuc22K = 1;
doSem22K = 0;
doSmallDB = 0;

% Which algorithms to run for finding hash fucntions
% doMLH: perform minimal loss hashing with hinge loss
% doBRE: perform minimal loss hashing with BRE loss, (our implementation of BRE)
% doLSH: perform locality sensitive hashing (LSH) which preserves cosine similarity
doMLH = 1;
doBRE = 0;
doLSH = 0;

% doPlots: whether to plot results (assuming MLH, BRE, LSH results are available)
doPlots = 0;


addpath utils;
addpath plots;

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if (doEuc22K) %%%%%%%%%%%%%%%%%%%%%%%%%% Euclidean 22K LabelMe %%%%%%%%%%%%%%%%%%%%%%%%%%
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  data = create_data('euc-22K-labelme', [], 0.01);
  perform_pca = 1			% whether to perform PCA dimensionality reduction

  if (perform_pca)
    data2 = do_pca(data, 40);
  else
    data2 = data;
  end

  if (doMLH) % ~~~~~~~~~~~~~~~~~~~~~~~~~ MLH with hinge loss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    clear Wtmp_rho Wtmp_shrink_w Wtmp_lambda W;
    clear pmlh rmlh best_params time_train time_validation;

    % verbose flag for validation is set to 0 (off)
    % to see the details during validation, set val_verbose to 15 (debug info every 15th iteration)
    val_verbose = 25;

    % in the paper n_trials = 10 used
    n_trials = 3;
    % different code lengths to try
    nbs = [16 32 64 128 256];

    for i = 1:n_trials
      for nb = nbs
        nb
        
        % ----- validation -----
        t0 = tic;
          
          % A heuristic for initial selection of rho
          data3 = create_training(data2, 'train', 1);
          [p0 r0] = eval_LSH(nb, data3);
          rho = sum(r0 < .3);			% rho with 30% recall (nothing deep; just a heuristic)
          fprintf('automatic estimation of rho suggested rho = %d.\n', rho);
          clear data3;
          
          % best_params is a data structure that stores the most reasonable parameter setting
          % initial setting of parameters
          best_params(i,nb).size_batches = 100;
          best_params(i,nb).eta = .1;
          best_params(i,nb).shrink_w = 1e-4;
          best_params(i,nb).lambda = .5;
          
          % number of learning iterations for validation
          val_iter = 75;
          % whether the hyper-plane offsets are zero during validation
          val_zerobias = 1;

          % validation for rho in hinge loss
          % rho might get large. If retrieval at a certain hamming distance at test time is desired, rho
          % should be set manually. See alternative validation on lambda (instead of rho) for small databases.  
          fprintf('validation for rho in hinge loss\n');    
          step = round(nb / 32);
          step(step < 1) = 1;
          
          rho_set = rho + [-2 -1 0 +1 +2] * step;
          rho_set(rho_set < 1) = [];
          Wtmp_rho = MLH(data2, {'hinge', rho_set, best_params(i,nb).lambda}, nb, [best_params(i,nb).eta], ...
                         .9, 100, 'train', val_iter, val_zerobias, 0, 5, val_verbose, best_params(i,nb).shrink_w, 0);
          best_ap = -1;
          for j = 1:numel(Wtmp_rho)
            if (Wtmp_rho(j).ap > best_ap)
              best_ap = Wtmp_rho(j).ap;
              best_params(i,nb).rho = Wtmp_rho(j).params.loss.rho;
            end
            if (val_verbose)
              fprintf('%.3f %d\n', Wtmp_rho(j).ap, Wtmp_rho(j).params.loss.rho);
            end
          end
          fprintf('Best rho (%d bits) = %d\n', nb, best_params(i,nb).rho);
          
          % validation for weight decay parameter
          fprintf('validation for weight decay parameter\n');
          shrink_w_set = [.01 1e-3 1e-4 1e-5 1e-6];
          Wtmp_shrink_w = MLH(data2, {'hinge', best_params(i,nb).rho, best_params(i,nb).lambda}, nb, ...
                              [best_params(i,nb).eta], .9, 100, 'train', val_iter, val_zerobias, 0, 5, ...
                              val_verbose, shrink_w_set, 0);
          best_ap = -1;
          for j = 1:numel(Wtmp_shrink_w)
            if (Wtmp_shrink_w(j).ap > best_ap)
              best_ap = Wtmp_shrink_w(j).ap;
              best_params(i,nb).shrink_w = Wtmp_shrink_w(j).params.shrink_w;
            end
            if (val_verbose)
              fprintf('%.0d %.6f\n', Wtmp_shrink_w(j).ap, Wtmp_shrink_w(j).params.shrink_w);
            end
          end
          fprintf('Best weight decay (%d bits) = %.0d\n', nb, best_params(i,nb).shrink_w);

          time_validation(i,nb) = toc(t0);  
          best_params(i,nb)

          % ---- training on the train+val set -----
          t1 = tic;
            % below 500 learning iterations are used, using more might provide slightly better results
            % in the paper we used 2000
            train_iter = 500;
            train_zerobias = 1;
            Wmlh{i, nb} = MLH(data2, {'hinge', best_params(i,nb).rho, best_params(i,nb).lambda}, nb, ...
                              [best_params(i,nb).eta], .9, [best_params(i,nb).size_batches], 'trainval', train_iter, train_zerobias, ...
                              5, 1, 50, [best_params(i,nb).shrink_w], 1);
            time_train(i,nb) = toc(t1);
            
            % computing precision / recall
            pmlh(i, nb, :) = zeros(1, max(nbs)+1);
            rmlh(i, nb, :) = zeros(1, max(nbs)+1);
            [pmlh(i, nb, 1:nb+1) rmlh(i, nb, 1:nb+1)] = eval_linear_hash(Wmlh{i, nb}.W, data2);
            
            if (perform_pca)
              save res/mlh_euc-22K-labelme-pca pmlh rmlh best_params Wmlh time_train time_validation;
            else
              save res/mlh_euc-22K-labelme pmlh rmlh best_params Wmlh time_train time_validation;
            end
      end
    end

  end
  if (doBRE) % ~~~~~~~~~~~~~~~~~~~~~~~~~~~ MLH with BRE loss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    % NOTE: The following script runs the implementation of Minimal Loss Hashing with BRE Cost
    % Function. The code of B. Kulis and T. Darrell, "Learning to Hash with Binary Reconstructive
    % Embeddings", NIPS 2009, can be downloaded from Brian Kulis's homepage. Their implementation uses a
    % different approach for optimization, i.e., co-ordinate descent.

    clear Wtmp_shrink_w Wbre;
    clear pbre rbre best_params time_train time_validation;

    % verbose flag for validation is set to 0 (off)
    % to see the details during validation set val_verbose to 10 (debug info every 10th iteration)
    val_verbose = 0;

    % in the paper results are reported for n_trials = 10.
    n_trials = 3;
    % different code lengths to try
    nbs = [16 32 64 128 256];

    for i = 1:n_trials
      for nb = nbs
        nb
        
        % ----- validation -----
        t0 = tic;

          % default setting of parameters
          best_params(i,nb).size_batches = 100;
          best_params(i,nb).eta = .1;
          
          % validation for weight decay parameter
          fprintf('validation for weight decay parameter\n');
          shrink_w_set = [.01 1e-3 1e-4 1e-5 1e-6];
          Wtmp_shrink_w = MLH(data2, {'bre'}, nb, [best_params(i,nb).eta], .9, 100, 'train', 75, 1, 0, ...
                              4, val_verbose, shrink_w_set, 0);
          best_ap = -1;
          for j = 1:numel(Wtmp_shrink_w)
            if (Wtmp_shrink_w(j).ap > best_ap)
              best_ap = Wtmp_shrink_w(j).ap;
              best_params(i,nb).shrink_w = Wtmp_shrink_w(j).params.shrink_w;
            end
            if (val_verbose)
              fprintf('%.3f %.6f\n', Wtmp_shrink_w(j).ap, Wtmp_shrink_w(j).params.shrink_w);
            end
          end
          fprintf('Best weight decay (%d bits) = %.0d\n', nb, best_params(i,nb).shrink_w);
          
          time_validation(i,nb) = toc(t0);
          
          best_params(i,nb)

          % ----- training on the train+val set -----
          t1 = tic;
            % below 500 learning iterations are used, using more might provide slightly better results
            Wbre{i, nb} = MLH(data2, {'bre'}, nb, [best_params(i,nb).eta], .9, [best_params(i,nb).size_batches], ...
                              'trainval', 500, 1, 5, 1, 50, [best_params(i,nb).shrink_w], 1);
            time_train(i,nb) = toc(t1);
            
            % computing precision / recall
            pbre(i, nb, :) = zeros(1, max(nbs)+1);
            rbre(i, nb, :) = zeros(1, max(nbs)+1);
            [pbre(i, nb, 1:nb+1) rbre(i, nb, 1:nb+1)] = eval_linear_hash(Wbre{i, nb}.W, data2);
            if (perform_pca)
              save res/bre_euc-22K-labelme-pca pbre rbre best_params Wbre time_train time_validation;
            else
              save res/bre_euc-22K-labelme pbre rbre best_params Wbre time_train time_validation;
            end  
      end
    end

  end
  if (doLSH) % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LSH ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    % in the paper results are reported for n_trials = 10.
    n_trials = 3;
    % different code lengths to try
    nbs = [16 32 64 128 256];

    clear plsh rlsh
    for i=1:n_trials
      fprintf('~~~ %d\n', i);
      for nb = nbs
        plsh(i, nb, :) = zeros(1, max(nbs)+1);
        rlsh(i, nb, :) = zeros(1, max(nbs)+1);
        [plsh(i, nb, 1:nb+1) rlsh(i, nb, 1:nb+1)] = eval_LSH(nb, data2);
      end
    end

    if (perform_pca)
      save res/lsh_euc-22K-labelme-pca plsh rlsh
    else
      save res/lsh_euc-22K-labelme plsh rlsh
    end

  end
  if (doPlots) % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PLOTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    load res/mlh_euc-22K-labelme-pca
    load res/bre_euc-22K-labelme-pca
    load res/lsh_euc-22K-labelme-pca

    nbs = [16 32 64 128 256];

    % Precision-Recall curves for different code lengths
    for nb = nbs;
      clear precs_mlh precs_lsh precs_bre; % precs_sh precs_sikh;
      
      n_mlh = sum(pmlh(:,nb,1) > 0);
      recs_mlh = [max(rmlh(1:n_mlh,nb,1)):.02:min(rmlh(1:n_mlh,nb,nb+1)), min(rmlh(1:n_mlh,nb,nb+1))];
      for i=1:n_mlh
        precs_mlh(i,:) = compute_prec_at_recall(squeeze(rmlh(i,nb,1:nb+1)), squeeze(pmlh(i,nb,1:nb+1)), recs_mlh);
      end

      n_lsh = sum(plsh(:,nb,1) > 0);
      recs_lsh = [max(rlsh(:,nb,1)):.02:min(rlsh(:,nb,nb+1)), min(rlsh(:,nb,nb+1))];
      for i=1:n_lsh
        precs_lsh(i,:) = compute_prec_at_recall(squeeze(rlsh(i,nb,1:nb+1)), squeeze(plsh(i,nb,1:nb+1)), recs_lsh);
      end

      n_bre = sum(pbre(:,nb,1) > 0);
      recs_bre = [max(rbre(:,nb,1)):.02:min(rbre(:,nb,nb+1)), min(rbre(:,nb,nb+1))];
      for i=1:size(pbre, 1)
        precs_bre(i,:) = compute_prec_at_recall(squeeze(rbre(i,nb,1:nb+1)), squeeze(pbre(i,nb,1:nb+1)), recs_bre);
      end
      
      % recs_sh = [max(rsh(:,nb,1)):.02:min(rsh(:,nb,nb+1)), min(rsh(:,nb,nb+1))];
      % for i=1:size(psh, 1)
      %   precs_sh(i,:) = compute_prec_at_recall(squeeze(rsh(i,nb,1:nb+1)), squeeze(psh(i,nb,1:nb+1)), recs_sh);
      % end
      
      cap.tit = ['22K Euc LabelMe (precision-recall) using ', num2str(nb), ' bits'];
      cap.xlabel = ['Recall'];
      cap.ylabel = ['Precision'];
      fig = make_err_plot({recs_mlh,            recs_bre,           recs_lsh,           }, ... % recs_sh,           }, ...
                          {mean(precs_mlh,1),   mean(precs_bre,1),  mean(precs_lsh,1),  }, ... % mean(precs_sh),    }, ...
                          {std(precs_mlh,0,1),  std(precs_bre,0,1), std(precs_lsh,0,1), }, ... % std(precs_sh,0,1), }, ...
                          {'MLH',               'BRE',              'LSH',              }, ... % 'SH',              }, ...	
                          cap, 'tr', 1);
      % exportfig(fig, ['figs/', 'Euc-22K-prec-recall-', num2str(nb), '.eps'], 'Color', 'rgb');
    end

  end % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
end % done with Euclidean 22K LabelMe ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if (doSem22K) %%%%%%%%%%%%%%%%%%%%%%%%%%% Semantic 22K LabelMe %%%%%%%%%%%%%%%%%%%%%%%%%%
              % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  data = create_data('sem-22K-labelme', 50, 1000);
  perform_pca = 0 % whether to perform PCA dimensionality reduction

  if (perform_pca)
    data2 = do_pca(data, 40);
  else
    data2 = data;
  end
  if (doMLH) % ~~~~~~~~~~~~~~~~~~~~~~~~~ MLH with hinge loss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    clear Wtmp_rho Wtmp_shrink_w Wtmp_lambda W;
    clear pmlh rmlh best_params time_train time_validation;

    % verbose flag for validation is set to 0 (off)
    % to see the details during validation, set val_verbose to 15 (debug info every 15th iteration)
    val_verbose = 0;

    % in the paper n_trials = 10 used
    n_trials = 1;
    % different code lengths to try
    nbs = [16 32 64 128 256];

    for i = 1:n_trials
      for nb = nbs
        nb

        % ----- validation -----
        t0 = tic;

          % A heuristic for initial selection of rho
          data3 = create_training(data2, 'train', 1);
          [p0 r0] = eval_LSH(nb, data3);
          rho = sum(r0 < .3);% rho with 30% recall (nothing deep; just a heuristic)
          fprintf('automatic estimation of rho suggested rho = %d.\n', rho);
          clear data3;

          % best_params is a data structure that stores the most reasonable parameter setting
          % initial setting of parameters
          best_params(i,nb).size_batches = 100;
          best_params(i,nb).eta = .1;
          best_params(i,nb).shrink_w = 1e-4;
          best_params(i,nb).lambda = .5;

          % number of learning iterations for validation
          val_iter = 75;
          % whether the hyper-plane offsets are zero during validation
          val_zerobias = 1;

          % validation for rho in hinge loss
          % rho might get large. If retrieval at a certain hamming distance at test time is desired, rho
          % should be set manually. See alternative validation on lambda (instead of rho) for small databases.
          fprintf('validation for rho in hinge loss\n');
          step = round(nb / 32);
          step(step < 1) = 1;

          rho_set = rho + [-2 -1 0 +1 +2] * step;
          rho_set(rho_set < 1) = [];
          Wtmp_rho = MLH(data2, {'hinge', rho_set, best_params(i,nb).lambda}, nb, [best_params(i,nb).eta], ...
                         .9, 100, 'train', val_iter, val_zerobias, 0, 5, val_verbose, best_params(i,nb).shrink_w, 0);
          best_ap = -1;
          for j = 1:numel(Wtmp_rho)
            if (Wtmp_rho(j).ap > best_ap)
              best_ap = Wtmp_rho(j).ap;
              best_params(i,nb).rho = Wtmp_rho(j).params.loss.rho;
            end
            if (val_verbose)
              fprintf('%.3f %d\n', Wtmp_rho(j).ap, Wtmp_rho(j).params.loss.rho);
            end
          end
          fprintf('Best rho (%d bits) = %d\n', nb, best_params(i,nb).rho);

          % validation for weight decay parameter
          fprintf('validation for weight decay parameter\n');
          shrink_w_set = [.01 1e-3 1e-4 1e-5 1e-6];
          Wtmp_shrink_w = MLH(data2, {'hinge', best_params(i,nb).rho, best_params(i,nb).lambda}, nb, ...
                              [best_params(i,nb).eta], .9, 100, 'train', val_iter, val_zerobias, 0, 5, ...
                              val_verbose, shrink_w_set, 0);
          best_ap = -1;
          for j = 1:numel(Wtmp_shrink_w)
            if (Wtmp_shrink_w(j).ap > best_ap)
              best_ap = Wtmp_shrink_w(j).ap;
              best_params(i,nb).shrink_w = Wtmp_shrink_w(j).params.shrink_w;
            end
            if (val_verbose)
              fprintf('%.3f %.6f\n', Wtmp_shrink_w(j).ap, Wtmp_shrink_w(j).params.shrink_w);
            end
          end
          fprintf('Best weight decay (%d bits) = %.0d\n', nb, best_params(i,nb).shrink_w);

          time_validation(i,nb) = toc(t0);
          best_params(i,nb)

          % ---- training on the train+val set -----
          t1 = tic;
            % blow 500 iterations used, using more iterations provides slightly better results
            % in the paper we used 2000
            train_iter = 500;
            train_zerobias = 1;
            Wmlh{i, nb} = MLH(data2, {'hinge', best_params(i,nb).rho, best_params(i,nb).lambda}, nb, ...
                              [best_params(i,nb).eta], .9, [best_params(i,nb).size_batches], 'trainval', train_iter, train_zerobias, ...
                              5, 1, 50, [best_params(i,nb).shrink_w], 1);
            time_train(i,nb) = toc(t1);

            % computing precision / recall
            pmlh(i, nb, :) = zeros(1, max(nbs)+1);
            rmlh(i, nb, :) = zeros(1, max(nbs)+1);
            [pmlh(i, nb, 1:nb+1) rmlh(i, nb, 1:nb+1)] = eval_linear_hash(Wmlh{i, nb}.W, data2);

            if (perform_pca)
              save res/mlh_sem-22K-labelme-pca pmlh rmlh best_params Wmlh time_train time_validation;
            else
              save res/mlh_sem-22K-labelme pmlh rmlh best_params Wmlh time_train time_validation;
            end
      end
    end

  end % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
end % done with Semantic 22K LabelMe ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if (doSmallDB) %%%%%%%%%%%%%%%%%%%%%%%%%%%% 6 Small Datasets %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
               % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  nbs = [10 15 20 25 30 35 40 45 50];
  
  n_trials = 3;  % number of trials for stochastic methods
                 % in the paper n_trials = 10 used

  if (doMLH) % ~~~~~~~~~~~~~~~~~~~~~~~~~~ MLH with hinge loss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    for modei = {'labelme', 'mnist', 'peekaboom', 'nursery', 'notredame', '10d'} 
      val_verbose = 0;
      train_verbose = 25;
      
      clear pmlh rmlh Wmlh;

      for i=1:n_trials
        mode = modei{1};
        fprintf('%s [%d / %d]\n', mode, i, n_trials);
        
        % raw data are available at http://www.eecs.berkeley.edu/~kulis/data/
        % re-create the datasets structure with new train/test subsets
        if (strcmp(mode, '10d'))
          data = create_data('uniform', 10);
        else
          data = create_data('kulis', mode);
        end
        % performs PCA dimentionality reduction to retain a 40D subspace
        if (strcmp(mode, 'nursery') || strcmp(mode, '10d'))
          data_pca = data;
        else
          data_pca = do_pca(data, 40);
        end;
        
        for nb = nbs
          % assumes rho = 3 / no validation on rho
          % Note that fixing rho is not the correct way of learning binary codes. Only use fixed rho if
          % all what you care about is the peformance quality at a specific rho
          rho = 3;
          % learning rate is fixed at .1 / no validation on eta
          eta = .1;
          
          fprintf('[nb = %d]\n', nb);
          t0 = tic;
            % % validation on lambda
            % lambda_set = [0 .2 .5];
            % Wtmp = MLH(data_pca, {'hinge', rho, lambda_set}, nb, [eta], .9, 100, 'train', 20, 1, 0, 5, ...
            % 		 val_verbose, 1e-4, 0);
            % [m ind] = max([Wtmp.ap]); % best setting according to evaluation
            % lambda = Wtmp(ind).params.loss.lambda;
            
            lambda = 0; % Because we fix rho at 3, we need to decrease lambda to act in higher precision
                        % regime. lambda being zero means that we don't resample positive pairs.
            
            % validation for the weight decay parameter
            shrink_w_set = [.01 1e-3 1e-4 1e-5];
            Wtmp = MLH(data_pca, {'hinge', rho, lambda}, nb, [eta], .9, 100, 'train', 20, 1, 0, 5, ...
                       val_verbose, shrink_w_set, 0);
            [m ind] = max([Wtmp.ap]); % best setting according to validation
            shrink_w = Wtmp(ind).params.shrink_w;
            time_validation(i, nb) = toc(t0);
            
            % training on the train+val set
            t1 = tic;
              Wmlh{i, nb} = MLH(data_pca, {'hinge', rho, lambda}, nb, [eta], .9, 100, 'trainval', 50, 1, ...
                                0, 1, train_verbose, shrink_w, 1);
              time_train(i, nb) = toc(t1);
        end
        
        for nb = nbs
          pmlh(i, nb, :) = zeros(1, 51);
          rmlh(i, nb, :) = zeros(1, 51);
          [pmlh(i, nb, 1:nb+1) rmlh(i, nb, 1:nb+1)] = eval_linear_hash(Wmlh{i, nb}.W, data_pca);
        end
        save(['res/mlh_', mode, '.mat'], 'pmlh', 'rmlh', 'mode', 'Wmlh', 'time_train', 'time_validation', 'n_trials');
      end
    end

  end
  if (doBRE) % ~~~~~~~~~~~~~~~~~~~~~~~~~~ MLH with BRE loss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    for modei = {'labelme', 'mnist', 'peekaboom', 'nursery', 'notredame', '10d'} 
      mode = modei{1}

      val_verbose = 0;
      train_verbose = 25;
      
      clear pbre rbre Wbre;
      for i=1:n_trials
        fprintf('[%d / %d]\n', i, n_trials);
        % raw data are available at http://www.eecs.berkeley.edu/~kulis/data/
        % re-create the datasets structure with new train/test subsets
        if (strcmp(mode, '10d'))
          data = create_data('uniform', 10);
        else
          data = create_data('kulis', mode);
        end
        % performs PCA dimentionality reduction to retain a 40D subspace
        if (strcmp(mode, 'nursery') || strcmp(mode, '10d'))
          data_pca = data;
        else
          data_pca = do_pca(data, 40);
        end;
        
        for nb = nbs
          % learning rate is fixed at .1 / no validation on eta
          eta = .1;
          
          fprintf('[nb = %d]\n', nb);
          t0 = tic;
            
            % validation for the weight decay parameter
            shrink_w_set = [.01 1e-3 1e-4 1e-5];
            Wtmp = MLH(data_pca, {'bre'}, nb, [eta], .9, 100, 'train', 20, 1, 0, 5, val_verbose, ...
                       shrink_w_set, 0);
            [m ind] = max([Wtmp.ap]); % best setting according to validation
            shrink_w = Wtmp(ind).params.shrink_w;
            time_validation(i, nb) = toc(t0);
            
            % training on the train+val set
            t1 = tic;
              Wbre{i, nb} = MLH(data_pca, {'bre'}, nb, [eta], .9, 100, 'trainval', 50, 1, 0, 1, train_verbose, ...
                                shrink_w, 1);
              time_train(i, nb) = toc(t1);
        end
        
        for nb = nbs
          pbre(i, nb, :) = zeros(1, 51);
          rbre(i, nb, :) = zeros(1, 51);
          [pbre(i, nb, 1:nb+1) rbre(i, nb, 1:nb+1)] = eval_linear_hash(Wbre{i, nb}.W, data_pca);
        end
        save(['res/bre_', mode, '.mat'], 'pbre', 'rbre', 'mode', 'Wbre', 'time_train', 'time_validation', 'n_trials');
      end
    end

  end
  if (doLSH) % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LSH ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    for modei = {'labelme', 'mnist', 'peekaboom', 'nursery', 'notredame', '10d'}
      mode = modei{1}
      
      clear pmlh rmlh W;
      for i=1:n_trials
        fprintf('[%d / %d]\n', i, n_trials);
        % raw data are available at http://www.eecs.berkeley.edu/~kulis/data/
        % re-create the datasets structure with new train/test subsets
        if (strcmp(mode, '10d'))
          data = create_data('uniform', 10);
        else
          data = create_data('kulis', mode);
        end
        % performs PCA dimentionality reduction to retain a 40D subspace
        if (strcmp(mode, 'nursery') || strcmp(mode, '10d'))
          data_pca = data;
        else
          data_pca = do_pca(data, 40);
        end;
        
        fprintf('~~~ %d\n', i);
        for nb = nbs
          plsh(i, nb, :) = zeros(1, max(nbs)+1);
          rlsh(i, nb, :) = zeros(1, max(nbs)+1);
          [plsh(i, nb, 1:nb+1) rlsh(i, nb, 1:nb+1)] = eval_LSH(nb, data_pca);
        end
      end
      
      save(['res/lsh_', mode, '.mat'], 'plsh', 'rlsh', 'mode', 'n_trials');
    end

  end
  if (doPlots) % ~~~~~~~~~~~~~~~~~~~~~~~ Plots for Small Datasets ~~~~~~~~~~~~~~~~~~~~~~~~~

    % Precision-Recall curves for different code lengths
    for modei = {'labelme', 'mnist', 'peekaboom', 'nursery', 'notredame', '10d'} 
      mode = modei{1}
      load(['res/mlh_', mode, '.mat']);
      load(['res/lsh_', mode, '.mat']);
      load(['res/bre_', mode, '.mat'])
      % load(['res/SH_', mode, '.mat']);
      
      for nb = [30, 50];
        clear precs_mlh precs_lsh precs_bre precs_lsh precs_sh precs_sikh;
        
        recs_mlh = [max(rmlh(:,nb,1)):.02:min(rmlh(:,nb,nb+1)), min(rmlh(:,nb,nb+1))];
        for i=1:size(pmlh, 1)
          precs_mlh(i,:) = compute_prec_at_recall(squeeze(rmlh(i,nb,1:nb+1)), squeeze(pmlh(i,nb,1:nb+1)), recs_mlh);
        end
        
        recs_bre = [max(rbre(:,nb,1)):.02:min(rbre(:,nb,nb+1)), min(rbre(:,nb,nb+1))];
        for i=1:size(pbre, 1)
          precs_bre(i,:) = compute_prec_at_recall(squeeze(rbre(i,nb,1:nb+1)), squeeze(pbre(i,nb,1:nb+1)), recs_bre);
        end
        
        recs_lsh = [max(rlsh(:,nb,1)):.02:min(rlsh(:,nb,nb+1)), min(rlsh(:,nb,nb+1))];
        for i=1:size(plsh, 1)
          precs_lsh(i,:) = compute_prec_at_recall(squeeze(rlsh(i,nb,1:nb+1)), squeeze(plsh(i,nb,1:nb+1)), recs_lsh);
        end
        
        % recs_sh = [max(rsh(:,nb,1)):.02:min(rsh(:,nb,nb+1)), min(rsh(:,nb,nb+1))];
        % for i=1:size(psh, 1)
        %   precs_sh(i,:) = compute_prec_at_recall(squeeze(rsh(i,nb,1:nb+1)), squeeze(psh(i,nb,1:nb+1)), recs_sh);
        % end

        cap.tit = [mode, ' (precision-recall) using ', num2str(nb), ' bits'];
        cap.xlabel = ['Recall'];
        cap.ylabel = ['Precision'];
        fig = make_err_plot({recs_mlh,    recs_bre,           recs_lsh,           }, ... % recs_sh,           }, ...
                            {mean(precs_mlh,1),   mean(precs_bre,1),  mean(precs_lsh,1),  }, ... % mean(precs_sh),    }, ...
                            {std(precs_mlh,0,1),  std(precs_bre,0,1), std(precs_lsh,0,1), }, ... % std(precs_sh,0,1), }, ...
                            {'MLH',               'BRE',              'LSH',              }, ... % 'SH',              }, ...	
                            cap, 'tr', 1);
        % exportfig(fig, ['figs/', mode, '-prec-recall-', num2str(nb), '.eps'], 'Color', 'rgb');
      end
    end

    % Precision (or Recall) at a certain Hamming distance R as a function of code length
    for R = [3];
      for modei = {'labelme', 'mnist', 'peekaboom', 'nursery', 'notredame', '10d'}
        
        mode = modei{1}
        
        load(['res/mlh_', mode, '.mat']);
        load(['res/lsh_', mode, '.mat']);
        load(['res/bre_', mode, '.mat'])
        % load(['res/SH_', mode, '.mat']);
        
        pmlh_std = squeeze(std(pmlh,0,1));
        pmlh_mean = squeeze(mean(pmlh,1));
        plsh_std = squeeze(std(plsh,0,1));
        plsh_mean = squeeze(mean(plsh,1));
        pbre_std = squeeze(std(pbre,0,1));
        pbre_mean = squeeze(mean(pbre,1));
        % psh_std = squeeze(std(psh,0,1));
        % psh_mean = squeeze(mean(psh,1));
        
        rmlh_std = squeeze(std(rmlh,0,1));
        rmlh_mean = squeeze(mean(rmlh,1));
        rlsh_std = squeeze(std(rlsh,0,1));
        rlsh_mean = squeeze(mean(rlsh,1));
        rbre_std = squeeze(std(rbre,0,1));
        rbre_mean = squeeze(mean(rbre,1));
        % rsh_std = squeeze(std(rsh,0,1));
        % rsh_mean = squeeze(mean(rsh,1));
        
        nbs_for_plot = [10 15 20 25 30 35 40 45 50];
        
        % how many models for each method
        % [size(pmlh,1) size(plsh,1) size(pbre,1)]
        
        cap.tit = [mode, ' (precision)'];
        cap.xlabel = ['Code length (bits)'];
        cap.ylabel = ['Precision for Hamm. dist. <= ', num2str(R)];
        p = [pmlh_mean(:,R+1) pbre_mean(:,R+1) plsh_mean(:,R+1)]; % psh_mean(:,R+1)];
        e = [pmlh_std(:,R+1)  pbre_std(:,R+1)  plsh_std(:,R+1) ]; % psh_std(:,R+1) ];
        n_lines = size(p,2);
        fig = make_err_plot(repmat(nbs_for_plot', [1 n_lines]), p(nbs_for_plot, :), e(nbs_for_plot, :), ...
                            {'MLH', 'BRE', 'LSH'}, ...% 'SH'}, ...
                            cap, 'br', 1);
        %  exportfig(fig, ['figs/',mode,'-prec-',num2str(R),'.eps'], 'Color', 'rgb');
        
        cap.tit = [mode, ' (recall)'];
        cap.xlabel = ['Number of bits'];
        cap.ylabel = ['Recall for Hamm. dist. <= ', num2str(R)];
        r =  [rmlh_mean(:,R+1) rbre_mean(:,R+1) rlsh_mean(:,R+1)]; % rsh_mean(:,R+1)];
        er = [rmlh_std(:,R+1)  rbre_std(:,R+1)  rlsh_std(:,R+1) ]; % rsh_std(:,R+1) ];
        n_lines = size(r,2);
        fig = make_err_plot(repmat(nbs_for_plot', [1 n_lines]), r(nbs_for_plot, :), er(nbs_for_plot, :), ...
                            {'MLH', 'BRE', 'LSH'}, ... % 'SH'}, ...
                            cap, 'tr', 1);
        %  exportfig(fig, ['figs/',mode,'-recall-',num2str(R),'.eps'], 'Color', 'rgb');
      end
    end

  end % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
end % done with small DBs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
