function [gaussian_model] = train_gmms(filelist, nGauss)
% ----------------------------------------------------------
% Function for training of isolated digit GMMs
%
% Your task is to fill in the missing parts of this function. 
% These sections are marked with "............."
%
% ----------------------------------------------------------
% (c) 2004 SpandH, Sheffield University
% ----------------------------------------------------------

%
% Start execution timer
%
tic;

%
% Model name definitions
%
strModels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'z', 'o'];
nModels = size(strModels, 2);

%
% Loading the data into the feature structure
%

ftrs = load_data(filelist);
ndim = size( ftrs{1} , 2 );

%
% train models for each digit ( store in models{i} ! )
%
for i = 1:nModels
  
    disp(['Creating GMM for ' strModels(i)]);
  
    % Create gmm structure (begin with a 1-mixture model)
    % for gaussian distributions with diagonal covariances
    gaussian_model(i) = gmm(ndim, 1, 'diag');
    % ............................................................................
    
     
    % Initialise models
    % add by Jiajie MA
    options = zeros(1,14);
    options(14) = 1000;
    gaussian_model(i) = gmminit(gaussian_model(i), ftrs{i}, options);
    % ............................................................................
    
    % Begin to increase the number of mixtures if neccessary

    for n = 2:nGauss

      % Mixup  
      % add by Jiajie MA
      gaussian_model(i) = gmmmixup(gaussian_model(i),n);
      iteration = 6;

      % ............................................................................
      x_list = [];
      y_list = [];
      % Do EM re-estimation  (use up to 6 iterations of EM training )
      for k=1:iteration

          gaussian_model(i) = gmmemstep(gaussian_model(i),ftrs{i});
          probability = sum(gmmlogprob(gaussian_model(i),ftrs{i}));
          y_list(k) = probability;
          x_list(k) =k;
          ylabel('Sum of Loglikelihood');
          xlabel('Iteration times');
          plot(x_list,y_list,'-');
          hold on;
      end
      % ............................................................................
      
    end
      %disp(ftrs{i})
      probability = gmmlogprob(gaussian_model(i),ftrs{i});
      %plot(x_list, probability,'linewidth', 2, 'markersize', 4);
      %most_likely = max(probability);
      %disp(['The most log likelihood is for label',num2str(i),' is:', num2str(most_likely)]);
      %disp(probability);
      %plot(probability, 'ko');
    
end

time = toc;
disp(['Elapsed training time = ', num2str(time),' secs']);
