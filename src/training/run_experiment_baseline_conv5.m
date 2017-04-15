function run_experiment_baseline_conv5(imdb_video)
%% Experiment entry point

	opts.gpus = 1;

	if nargin < 1
	    imdb_video = [];
	end

    opts.join.method = 'xcorr';
    opts.branch.conf.num_out = [96 256 384 384 32];
    opts.branch.conf.num_in = [ 3 48 256 192 192];
    opts.branch.conf.conv_stride = [ 2 1 1 1 1];
    opts.branch.conf.pool_stride = [ 2 1];
 
    opts.exemplarSize = 127;
   	opts.train.numEpochs = 100;

	experiment(imdb_video, opts);

end

