function [ stats ] = eval_gmms( models, filelist )
% -----------------------------------------------------------------
% Function for evaluating the performance on a test set
%
% Your task is to fill in the missing parts of this function. These
% sections will be marked with "..."
% 
% -----------------------------------------------------------------

%
% Start execution timer
%
tic;

%
% Load the file names
%
[fid,mesg] = fopen(filelist,'r');
if fid<0; disp(mesg); end
fnames = fscanf(fid,'%c');
fclose(fid);

%
% Model name definitions
%
strModels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'z', 'o'];
nModels = size(strModels, 2);

%
% Initialise scoring statistic
%
stats = newStats(length(strModels));

%
% Loop over all isolated digit files
%
[fn,fnames]=strtok(fnames);
fid = fopen('evalresult.txt','wt');
while ~isempty(fn)

  %
  % Load the features for the current test sample
  %
  len = length(fn); 
  place = find(fn == '.'); 
  ref = fn(place -2);
  ftrs = readhtk(fn);
  %disp(ref);
  %
  % Get the most likely model, given the observations
  % 
  % store the result string in "test"
  % ................................................................................
  % Created by Jiajie MA
  for i=1:nModels
     probability = gmmlogprob(models(i),ftrs);
     total_probability(i) = sum(probability);
  end
  
  % Created by Jiajie MA
  [max_probability,test] = max(total_probability);
  test = num2str(test);
  if test == '10'
      test = 'z';
  end
  if test == '11'
      test = 'o';
  end
  % ................................................................................

  %
  % Compare test and reference labels.
  %
  res = recognitionStats(test, ref, strModels);
  stats = accumulateStats(stats, res);
  %Create file add by Jiajie Ma
  
  fprintf(fid,'%s','<filename-');
  fprintf(fid,'%s',ref);
  fprintf(fid,'%s','>');
  fprintf(fid,'%s','<result-');
  fprintf(fid,'%s',test);
  fprintf(fid,'%s\n','>');
  
  %
  % get the next file
  [fn,fnames]=strtok(fnames);
end
fprintf(fid,'%s','The accuracy is:');
fprintf(fid,'%s',num2str(stats.acc));
fclose(fid);
disp(['Accuracy= ' num2str(stats.acc) ' [%]']);
time = toc;
disp(['Elapsed evaluation time = ', num2str(time),' secs']);