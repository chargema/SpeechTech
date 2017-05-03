function hmm = viterbi_train(hmm, traindata, bNeedInit);
% ----------------------------------------------------------
% hmm =  viterbi_train( hmm , traindata , bNeedInit );
%
% bNeedInit is 1 if we need to initialise the hmm, 
% otherwise. Default 1.
%
% ----------------------------------------------------------
% (c) 2004 SpandH, Sheffield University
% ----------------------------------------------------------

if nargin < 3
  bNeedInit = 1;
end
  
% the number of training data
ntrain = length(traindata);
nstates = hmm.nstates;

%
% Initialise the HMM
% 
statedata = cell(1,nstates);
if bNeedInit,
  
  %
  % compute the global variance for flooring
  %

  psi = cell(1,ntrain);
  for r=1:ntrain,
    %
    % uniform segmentation of the data
    %
    nobs=size(traindata{r},1);
    psi{r} = uniSegment(nstates, nobs);

    %
    % Collecting data into array
    %
    for j =1:nstates
      statedata{j} = [statedata{j}; traindata{r}(find(psi{r}(j,:)==1),:)];
    end
  end
  ndim = size(statedata{1}, 2);
  
  %
  % compute the global variance for flooring
  %
  alldata = statedata{1};
  for j = 2:nstates
    alldata = [ alldata ; statedata{j} ];
  end
  hmm.varfloor = 0.01 * std(alldata).^2;
  
  %
  % initialise HMM states 
  %
  for j = 1:nstates
    hmm.gmms{j} = gmm(ndim, 1, 'diag');
    hmm.gmms{j}.centres = mean(statedata{j});
    hmm.gmms{j}.covars = std(statedata{j}).^2;
  end
  
  % 
  % Initialise the transition matrix by setting all probs to 0.5 
  %
  hmm.pself(:) = log(0.5);
  hmm.pnext(:) = log( 1-exp(hmm.pself(:)) );
  hmm.pself(nstates) = log(1.0);
  hmm.pnext(nstates) = -Inf;
  
end

%
% Perform 3 iterations of Viterbi training
%
niter = 3;
for iteration = 1:niter,

  %
  % initialise the data structures to collect the
  % data assigned to each state
  %
  statedata = cell(1,nstates);
  
  %
  % Loop over all training samples
  %
  vi_prob = zeros(length(traindata),1);
  for r = 1:ntrain,
    %
    % Viterbi segmentation
    %
    
    % ............................................................................
    [stateseq,prob] = viterbi_path(hmm,traindata{r}); % compute the best path and whose prob
    vi_prob(r) = prob;
    %nobs=size(traindata{r},1);
    %statefrq = uniSegment(nstates, nobs);
    %stateseq = zeros(1,size(statefrq,2));
    %for x=1:size(statefrq,1)
     %   for y=1:size(statefrq,2)
      %      if statefrq(x,y) == 1
       %        stateseq(y) = x;
        %    end
        %end
    %end
                
    %
    % Now each data frame has been assigned to a state!
    %
    % Collect all the frames in a matrix for each state
    % 

    % ............................................................................
    
    %
    % Accumulate counts for the transition matrix
    %
    % HINT: you only need to count the number of self transitions 
    % for each state due (this is because we have left-to-right HMMs
    %
    nself = zeros(1,1);% add by Jiajie MA, initialise the nself
    nself = nself + diff([1,find(diff(stateseq)),length(stateseq)])-1;
    % ............................................................................
  end  
  %
  % now train each states individually by using GMM EM training
  %
  % Perform 1-6 iterations of EM at each stage
  %
  % add by Jiajie MA
  for i = 1:nstates
      iteration = 6;
      for k=1:iteration
          hmm.gmms{i} = gmmemstep(hmm.gmms{i},traindata{i});
      end
  end
  % ............................................................................
  
  % 
  % Re-estimate the transition probabilities
  %
  % NOTE: they are stored as log-probabilities, so take the log 
  % after computing them
  %
  % ............................................................................
  maxp = max(vi_prob);
  %disp(maxp);
  %disp(stateseq);
  % add by Jiajie MA
  for i = 1:nstates
      if nself(i) == -1
          nself(i) = nself(i)+1;
      end
      hmm.pself(i) = (nself(i)/sum(nself));
      hmm.pnext(i) = (1-(hmm.pself(i)));
      %hmm.pself(i) = log(hmm.pself(i));
      %hmm.pnext(i) = log(hmm.pnext(i));
  end
  % 
  % check if covariances are too small
  %

  for j = 1:nstates
    for m = 1:hmm.gmms{j}.ncentres,
      k = ( hmm.gmms{j}.covars(m,:) < hmm.varfloor );
      hmm.gmms{j}.covars(m,k) = hmm.varfloor(k);
    end
  end
  
end % iterations