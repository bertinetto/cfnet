
% -------------------------------------------------------------------------------------------------
function [net, stats] = experiment(imdb_video, varargin)
% Experiment main function - creates a network and trains it on the dataset.
% -------------------------------------------------------------------------------------------------
    % Default parameters (set the experiment-specific ones in run_experiment)
    opts.branch.type = 'alexnet';
    opts.branch.conf = struct(); % Options depend on type of net.
    % Do not add options to this struct here.
    % (vl_argparse does not expect fields that are not here)
    opts.join.method = 'xcorr'; % {xcorr, corrfilt}
    opts.join.conf = struct(); % Options depend on method.
    % Do not add options to this struct here.
    % (vl_argparse does not expect fields that are not here)
    opts.share_params = true;
    opts.pretrain = false; % Location of model file set in env_paths.
    opts.init.scale = 1;
    opts.init.weightInitMethod = 'xavierimproved';
    opts.init.initBias = 0.1;
    opts.expDir = 'data'; % where to save the trained net
    opts.numFetchThreads = 12; % used by vl_imreadjpg when reading dataset
    opts.exemplarSize = 127; % training image (use 255 for corrfilt, 127 for xcorr)
    opts.instanceSize = 255; % - 2*8; % search region
    opts.loss.type = 'simple';
    opts.loss.rPos = 8; % pixel with distance from center d > rPos are given a negative label
    opts.loss.rNeg = 0; % if rNeg != 0 pixels rPos < d < rNeg are given a neutral label
    opts.loss.labelWeight = 'balanced';
    opts.loss.cost = 'dist'; % {iou, dist}
    opts.loss.max_dist = inf; % {iou, dist}
    opts.pairsPerVideoTrain =  25; % avg number of pairs per training video (3862)
    opts.pairsPerVideoVal =  20; % avg number of pairs per validation video (555)
    opts.negatives = 0; % fraction of negative pairs taken from different videos
    opts.hardNegatives = 0; % fraction of negative pairs taken from same video
    opts.randomSeed = 0;
    opts.shuffleDataset = false; % do not shuffle the data to get reproducible experiments
    opts.frameRange = 100; % range from the exemplar in which randomly pick the instance
    opts.gpus = [];
    opts.prefetch = false; % Both get_batch and cnn_train_dag depend on prefetch.
    opts.train.numEpochs = 50;
    opts.train.learningRate = logspace_len(-2, -4, 50, opts.train.numEpochs);
    opts.train.weightDecay = 5e-4;
    opts.train.batchSize = 8; % we empirically observed that small batches work better
    opts.train.numSubBatches = 1;
    opts.train.profile = false;
    % Data augmentation settings
    opts.subMean = false;
    opts.colorRange = 255;
    opts.augment.translate = false;
    opts.augment.maxTranslate = 4;
    opts.augment.stretch = false;
    opts.augment.maxStretch = 0.05;
    opts.augment.color = true;
    opts.augment.grayscale = 0; % likelihood of using grayscale pair
    % Override default parameters if specified in run_experiment
    opts = vl_argparse(opts, varargin);
    % Get environment-specific default paths.
    opts = env_paths_training(opts);
    opts.train.gpus = opts.gpus;
    opts.train.prefetch = opts.prefetch;
% -------------------------------------------------------------------------------------------------
    % Get ImageNet Video metadata
    if isempty(imdb_video)
        fprintf('loading imdb video...\n');
        imdb_video = load(opts.imdbVideoPath);
        imdb_video = imdb_video.imdb_video;
    end

    % Load dataset statistics for data augmentation
    [rgbMean_z, rgbVariance_z, rgbMean_x, rgbVariance_x] = load_stats(opts);
    if opts.shuffleDataset
        s = RandStream.create('mt19937ar', 'Seed', 'shuffle');
        opts.randomSeed = s.Seed;
    end

    opts.train.expDir = opts.expDir;

    rng(opts.randomSeed); % Re-seed before calling make_net.

    % -------------------------------------------------------------------------------------------------
    net = make_net(opts);
    print_net_graph(net, opts);
    % -------------------------------------------------------------------------------------------------

    [imdb_video, imdb] = choose_val_set(imdb_video, opts);

    [resp_sz, resp_stride] = get_response_size(net, opts);
    [tmpl_sz, tmpl_stride] = get_template_size(net, opts);
    [net, derOutputs, label_inputs_fn] = setup_loss(net, resp_sz, resp_stride, opts.exemplarSize, opts.instanceSize, opts.loss);

    batch_fn = @(db, batch) get_batch(db, batch, ...
        imdb_video, ...
        opts.rootDataDir, ...
        numel(opts.train.gpus) >= 1, ...
        struct('exemplarSize', opts.exemplarSize, ...
               'instanceSize', opts.instanceSize, ...
               'frameRange', opts.frameRange, ...
               'negatives', opts.negatives, ...
               'hardNegatives', opts.hardNegatives, ...
               'loss', opts.loss.type, ...
               'subMean', opts.subMean, ...
               'colorRange', opts.colorRange, ...
               'stats', struct('rgbMean_z', rgbMean_z, ...
                               'rgbVariance_z', rgbVariance_z, ...
                               'rgbMean_x', rgbMean_x, ...
                               'rgbVariance_x', rgbVariance_x), ...
               'augment', opts.augment, ...
               'prefetch', opts.train.prefetch, ...
               'numThreads', opts.numFetchThreads), ...
        label_inputs_fn, ...
        opts.join.method);

    expm_folder = strsplit(fileparts(pwd), '/');
    post_epoch_fn = @(ep) otb_evaluation(ep, opts.epochs_to_test, opts.expDir, opts.join.method, resp_sz(1), resp_stride, opts.gpus);
    print_err_fn = @(ep,stats) print_test_err(ep, stats, expm_folder{end}, opts.epochs_to_test);
    opts.train.derOutputs = derOutputs;
    opts.train.randomSeed = opts.randomSeed;
    % -------------------------------------------------------------------------------------------------
    % Start training
    [net, stats] = cnn_train_dag(net, imdb, batch_fn, opts.train);
    % -------------------------------------------------------------------------------------------------
end


% -----------------------------------------------------------------------------------------------------
function [rgbMean_z, rgbVariance_z, rgbMean_x, rgbVariance_x] = load_stats(opts)
% Dataset image statistics for data augmentation
% -----------------------------------------------------------------------------------------------------
    stats = load(opts.imageStatsPath);
    % Subtracted if opts.subMean is true
    if ~isfield(stats, 'z')
        rgbMean = reshape(stats.rgbMean, [1 1 3]);
        rgbMean_z = rgbMean;
        rgbMean_x = rgbMean;
        [v,d] = eig(stats.rgbCovariance);
        rgbVariance_z = 0.1*sqrt(d)*v';
        rgbVariance_x = 0.1*sqrt(d)*v';
    else
        rgbMean_z = reshape(stats.z.rgbMean, [1 1 3]);
        rgbMean_x = reshape(stats.x.rgbMean, [1 1 3]);
        % Set data augmentation statistics, used if opts.augment.color is true
        [v,d] = eig(stats.z.rgbCovariance);
        rgbVariance_z = 0.1*sqrt(d)*v';
        [v,d] = eig(stats.x.rgbCovariance);
        rgbVariance_x = 0.1*sqrt(d)*v';
    end
end

% -------------------------------------------------------------------------------------------------
function print_net_graph(net, opts)
% -------------------------------------------------------------------------------------------------
    % Save the net graph to disk.
    inputs = {'exemplar', [opts.exemplarSize*[1 1] 3 opts.train.batchSize], ...
              'instance', [opts.instanceSize*[1 1] 3 opts.train.batchSize]};
    net_dot = net.print(inputs, 'Format', 'dot');
    if ~exist(opts.expDir)
        mkdir(opts.expDir);
    end

    dot_file = fullfile(opts.expDir, 'arch.dot');
    pdf_file = fullfile(opts.expDir, 'arch.pdf');
    f = fopen(dot_file, 'w');
    fprintf(f, net_dot);
    fclose(f);
    [~, ~] = system(['dot -Tpdf ' dot_file ' >' pdf_file]);
end


% -------------------------------------------------------------------------------------------------
function [resp_sz, resp_stride] = get_response_size(net, opts)
% -------------------------------------------------------------------------------------------------
    sizes = net.getVarSizes({'exemplar', [opts.exemplarSize*[1 1] 3 256], ...
                             'instance', [opts.instanceSize*[1 1] 3 256]});
    resp_sz = sizes{net.getVarIndex('score')}(1:2);
    rfs = net.getVarReceptiveFields('exemplar');
    resp_stride = rfs(net.getVarIndex('score')).stride(1);
    assert(all(rfs(net.getVarIndex('score')).stride == resp_stride));
end

% -------------------------------------------------------------------------------------------------
function [tmpl_sz, tmpl_stride] = get_template_size(net, opts)
% -------------------------------------------------------------------------------------------------
    sizes = net.getVarSizes({'exemplar', [opts.exemplarSize*[1 1] 3 256], ...
                             'instance', [opts.instanceSize*[1 1] 3 256]});
    tmpl_sz = sizes{net.getVarIndex('br1_out')}(1:2);
    rfs = net.getVarReceptiveFields('exemplar');
    tmpl_stride = rfs(net.getVarIndex('br1_out')).stride(1);
    assert(all(rfs(net.getVarIndex('br1_out')).stride == tmpl_stride));
end


% -------------------------------------------------------------------------------------------------
function [net, derOutputs, inputs_fn] = setup_loss(net, resp_sz, resp_stride, crop_sz_z, crop_sz_x, loss_opts)
% Add layers to the network and constructs a function that returns the inputs required by the loss layer.
% -------------------------------------------------------------------------------------------------

    switch loss_opts.type
        case 'simple'
            %% create label and weights for logistic loss
            net.addLayer('objective', ...
                         LogisticLoss(), ...
                         {'score', 'eltwise_label', 'eltwise_weight'}, 'objective');
            % adding weights to loss layer
            [pos_eltwise, neg_eltwise, pos_weight, neg_weight] = create_labels(...
                resp_sz, loss_opts.labelWeight, ...
                loss_opts.rPos/resp_stride, loss_opts.rNeg/resp_stride);

            derOutputs = {'objective', 1};
            inputs_fn = @(labels, obj_sz_z, obj_sz_x) get_label_inputs_simple(...
                labels, obj_sz_z, obj_sz_x, pos_eltwise, neg_eltwise, pos_weight, neg_weight);

        otherwise
            error('Unknown loss')
    end

    switch loss_opts.type
        case 'simple'
            net.addLayer('errdisp', centerThrErr('stride', resp_stride), ...
                         {'score','label'}, 'errdisp');
            net.addLayer('iou', IOUErrorScore('stride', resp_stride), ...
                         {'score', 'label', 'exemplar_size', 'instance_size'}, ...
                         'iou');
        otherwise
            error('Unknown loss')
    end
end

% -------------------------------------------------------------------------------------------------
function inputs = get_label_inputs_simple(labels, obj_sz_z, obj_sz_x, pos_eltwise, neg_eltwise, wp_eltwise, wn_eltwise)
% -------------------------------------------------------------------------------------------------
    pos = (labels > 0);
    neg = (labels < 0);

    resp_sz = size(pos_eltwise);
    eltwise_labels = zeros([resp_sz, 1, numel(labels)], 'single');
    eltwise_labels(:,:,:,pos) = repmat(pos_eltwise, [1 1 1 sum(pos)]);
    eltwise_labels(:,:,:,neg) = repmat(neg_eltwise, [1 1 1 sum(neg)]);
    eltwise_weights = zeros([resp_sz, 1, numel(labels)], 'single');
    eltwise_weights(:,:,:,pos) = repmat(wp_eltwise, [1 1 1 sum(pos)]);
    eltwise_weights(:,:,:,neg) = repmat(wn_eltwise, [1 1 1 sum(neg)]);
    inputs = {'label', labels, ...
              'eltwise_label', eltwise_labels, ...
              'eltwise_weight', eltwise_weights, ...
              'exemplar_size', obj_sz_z, ...
              'instance_size', obj_sz_x};
end

% -------------------------------------------------------------------------------------------------
function bbox = bbox_from_crop(crop_sz, obj_sz, batch_size)
% -------------------------------------------------------------------------------------------------
    % x, y, w, h
    bbox = zeros(4, batch_size);
    bbox(1:2,:) = crop_sz/2 - [obj_sz(2,:); obj_sz(1,:)]/2;
    bbox(3:4,:) = [obj_sz(2,:); obj_sz(1,:)];
    bbox = reshape(bbox,[1 1 4 batch_size]); % reshape it as tensor
end

% -------------------------------------------------------------------------------------------------
function [imdb_video, imdb] = choose_val_set(imdb_video, opts)
% Designates some examples for validation.
% It modifies imdb_video and constructs a dummy imdb.
% -------------------------------------------------------------------------------------------------
    TRAIN_SET = 1;
    VAL_SET = 2;

    nt = sum(imdb_video.set==TRAIN_SET);
    nv = sum(imdb_video.set==VAL_SET);
    num_pairs_train = round(nt * opts.pairsPerVideoTrain);
    num_pairs_val = round(nv * opts.pairsPerVideoVal);
    num_pairs = num_pairs_train + num_pairs_val;
    %% create imdb of indexes to imdb_video
    % train and val from disjoint video sets
    imdb = struct();
    imdb.images = struct(); % we keep the images struct for consistency with cnn_train_dag (MatConvNet)
    imdb.id = 1:num_pairs;
    imdb.images.set = uint8(zeros(1, num_pairs));
    imdb.images.set(1:num_pairs_train) = TRAIN_SET;
    imdb.images.set(num_pairs_train+1:end) = VAL_SET;
end


% -------------------------------------------------------------------------------------------------
function inputs = get_batch(db, batch, imdb_video, data_dir, use_gpu, sample_opts, label_inputs_fn, join_method)
% Returns the inputs to the network.
% -------------------------------------------------------------------------------------------------

    [imout_z, imout_x, labels, sizes_z, sizes_x] = vid_get_random_batch(...
        db, imdb_video, batch, data_dir, sample_opts);
    if use_gpu
        imout_z = gpuArray(imout_z);
        imout_x = gpuArray(imout_x);
    end
    % Constructs full label inputs from output of vid_get_random_batch.
    label_inputs = label_inputs_fn(labels, sizes_z, sizes_x);
    inputs = [{'exemplar', imout_z, 'instance', imout_x}, label_inputs];
end
