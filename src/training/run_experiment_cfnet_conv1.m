function run_experiment_cfnet_conv1(imdb_video)
%% Experiment entry point

    startup;
    opts.gpus = 1;
    if nargin < 1
        imdb_video = [];
    end

    opts.join.method = 'corrfilt';
    opts.join.conf.lambda = 10;
    opts.join.conf.window = 'cos';
    opts.join.conf.sigma = 8;
    opts.join.conf.target_lr = 0;

    opts.branch.conf.last_layer = 'relu1';
    opts.branch.conf.num_out     = [32];
    opts.branch.conf.num_in      = [ 3 ];
    opts.branch.conf.conv_stride = [ 4];
    opts.branch.conf.pool_stride = [ 2];

    opts.exemplarSize = 255;
    opts.train.numEpochs = 100;

    experiment(imdb_video, opts);
end

