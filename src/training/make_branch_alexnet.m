function net = make_branch_alexnet_nodilation(varargin)
    opts.exemplarSize = [127 127];
    opts.instanceSize = [255 255];
    opts.last_layer = 'conv5';
    opts.num_out     = [96, 256, 384, 384, 256];
    opts.num_in      = [ 3,  48, 256, 192, 192];
    opts.conv_stride = [ 2,   1,   1,   1,   1];
    opts.pool_stride = [ 2,   2];
    opts.scale = 1 ;
    opts.initBias = 0.1 ;
    opts.weightDecay = 1 ;
    opts.weightInitMethod = 'gaussian';
    opts.batchNormalization = false ;
    opts.cudnnWorkspaceLimit = 1024*1024*1024 ; % 1GB
    opts = vl_argparse(opts, varargin) ;

    if numel(opts.exemplarSize) == 1
        opts.exemplarSize = [opts.exemplarSize, opts.exemplarSize];
    end
    if numel(opts.instanceSize) == 1
        opts.instanceSize = [opts.instanceSize, opts.instanceSize];
    end

    net = struct();

    net.layers = {} ;

    for i = 1:numel(opts.num_out)
        switch i
            case 1
                net = add_block(net, opts, '1', 11, 11, ...
                                opts.num_in(i), opts.num_out(i), ...
                                opts.conv_stride(i), 0, 1) ;
                net = add_norm(net, opts, '1') ;
                net.layers{end+1} = struct('type', 'pool', 'name', 'pool1', ...
                                           'method', 'max', ...
                                           'pool', [3 3], ...
                                           'stride', opts.pool_stride(i), ...
                                           'pad', 0) ;
            case 2
                net = add_block(net, opts, '2', 5, 5, ...
                                opts.num_in(i), opts.num_out(i), ...
                                opts.conv_stride(i), 0, 1) ;
                net = add_norm(net, opts, '2') ;
                net.layers{end+1} = struct('type', 'pool', 'name', 'pool2', ...
                                           'method', 'max', ...
                                           'pool', [3 3], ...
                                           'stride', opts.pool_stride(i), ...
                                           'pad', 0) ;
            case 3
                net = add_block(net, opts, '3', 3, 3, ...
                                opts.num_in(i), opts.num_out(i), ...
                                opts.conv_stride(i), 0, 1) ;

%                 net = add_dropout(net, opts, '3');
            case 4
                net = add_block(net, opts, '4', 3, 3, ...
                                opts.num_in(i), opts.num_out(i), ...
                                opts.conv_stride(i), 0, 1) ;
%                 net = add_dropout(net, opts, '4');                            
            case 5
                net = add_block(net, opts, '5', 3, 3, ...
                                opts.num_in(i), opts.num_out(i), ...
                                opts.conv_stride(i), 0, 1) ;
        end
    end

    ind = find(cellfun(@(l) strcmp(l.name, opts.last_layer), net.layers));
    if numel(ind) ~= 1
        error(sprintf('could not find one layer: %s', opts.last_layer));
    end
    net.layers = net.layers(1:ind);

    % Check if the receptive field covers full image

    [ideal_exemplar, ~] = ideal_size(net, opts.exemplarSize);
    [ideal_instance, ~] = ideal_size(net, opts.instanceSize);
    assert(sum(opts.exemplarSize==ideal_exemplar)==2, 'exemplarSize is not ideal.');
    assert(sum(opts.instanceSize==ideal_instance)==2, 'instanceSize is not ideal.');

    % Fill in default values
    net = vl_simplenn_tidy(net) ;

end

% --------------------------------------------------------------------
function net = add_block(net, opts, id, h, w, in, out, stride, pad, dilate)
% --------------------------------------------------------------------
    convOpts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
    net.layers{end+1} = struct('type', 'conv', 'name', sprintf('conv%s', id), ...
                               'weights', {{init_weight(opts, h, w, in, out, 'single'), ...
                                            zeros(out, 1, 'single')}}, ...
                               'stride', stride, ...
                               'pad', pad, ...
                               'dilate', dilate, ...
                               'learningRate', [1 2], ...
                               'weightDecay', [opts.weightDecay 0], ...
                               'opts', {convOpts}) ;
    if opts.batchNormalization
        net.layers{end+1} = struct('type', 'bnorm', 'name', sprintf('bn%s',id), ...
                                   'weights', {{ones(out, 1, 'single'), ...
                                                zeros(out, 1, 'single'), ...
                                                zeros(out, 2, 'single')}}, ...
                                   'learningRate', [2 1 0.05], ...
                                   'weightDecay', [0 0]) ;
    end
    net.layers{end+1} = struct('type', 'relu', 'name', sprintf('relu%s',id)) ;
end

% --------------------------------------------------------------------
function net = add_norm(net, opts, id)
% --------------------------------------------------------------------
    if ~opts.batchNormalization
      net.layers{end+1} = struct('type', 'normalize', ...
                                 'name', sprintf('norm%s', id), ...
                                 'param', [5 1 0.0001/5 0.75]) ;
    end
end

% --------------------------------------------------------------------
function net = add_dropout(net, opts, id)
% --------------------------------------------------------------------
    net.layers{end+1} = struct('type', 'dropout', ...
                                 'name', sprintf('dropout%s', id), ...
                                 'rate', 0.5) ;
end
