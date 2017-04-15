function net = make_net(opts)
% MAKE_NET constructs the network up to the output.
% Loss functions are not added.
%
% Options include:
% Architecture to use for initial branches of network.
% Whether to use scores or bounding box regression.
% Whether to use xcorr or concat layer to join.
% Architecture to put after concat (1x1 convs).

switch opts.branch.type
    case 'alexnet'
        [branch1, branch2] = make_branches_alexnet(opts);
    otherwise
        error('Unknown branch type')
end
[repr_sz, repr_stride] = output_size(branch1, [opts.exemplarSize*[1 1], 3, 1]);
switch opts.join.method
    case 'xcorr'
        [join, final] = make_join_xcorr(opts);
    case 'corrfilt'
        [join, final] = make_join_corr_filt(opts, repr_sz, repr_stride);
    otherwise
        error('Unknown join method')
end

net = make_siamese(branch1, branch2, join, final, ...
                   {'exemplar', 'instance'}, 'score', ...
                   'share_all', opts.share_params);

end

function [branch1, branch2] = make_branches_alexnet(opts)
    branch_opts.last_layer = 'conv5';
    branch_opts.num_out     = [96, 256, 384, 384, 256];
    branch_opts.num_in      = [ 3,  48, 256, 192, 192];
    branch_opts.conv_stride = [ 2,   1,   1,   1,   1];
    branch_opts.pool_stride = [ 2,   2];
    branch_opts.batchNormalization = true;
    branch_opts = vl_argparse(branch_opts, {opts.branch.conf});

    branch_opts.exemplarSize = opts.exemplarSize * [1 1];
    branch_opts.instanceSize = opts.instanceSize * [1 1];
    branch_opts.weightInitMethod = opts.init.weightInitMethod;
    branch_opts.scale            = opts.init.scale;
    branch_opts.initBias         = opts.init.initBias;

    f = @() make_branch_alexnet(branch_opts);
    branch1 = f();
    branch2 = f();
end


function [join, final] = make_join_xcorr(opts)
    join_opts.finalBatchNorm = true;
    join_opts.adjustGainInit = 1;
    join_opts.adjustBiasInit = 0;
    % Learning rates ignored if batch-norm is enabled.
    join_opts.adjustGainLR = 0;
    join_opts.adjustBiasLR = 1;
    join_opts = vl_argparse(join_opts, {opts.join.conf});

    join = dagnn.DagNN();
    join.addLayer('xcorr', XCorr(), {'in1', 'in2'}, {'out'});

    % Create adjust layer.
    final.layers = {};
    convOpts = {'CudnnWorkspaceLimit', 1024*1024*1024};
    if join_opts.finalBatchNorm
        % Batch-norm layer only.
        final.layers{end+1} = struct(...
            'type', 'bnorm', 'name', 'adjust_bn', ...
            'weights', {{single(join_opts.adjustGainInit), ...
                         single(join_opts.adjustBiasInit), ...
                         zeros(1, 2, 'single')}}, ...
            'learningRate', [2 1 0.3], ...
            'weightDecay', [0 0]);
    else
        % Linear layer only.
        final.layers{end+1} = struct(...
            'type', 'conv', 'name', 'adjust', ...
            'weights', {{single(join_opts.adjustGainInit), ...
                         single(join_opts.adjustBiasInit)}}, ...
            'learningRate', [join_opts.adjustGainLR, join_opts.adjustBiasLR], ...
            'weightDecay', [1 0], ...
            'opts', {convOpts});
    end
end

function [join, final] = make_join_corr_filt(opts, in_sz, in_stride)
    join_opts.finalBatchNorm = false;
    join_opts.const_cf = false;
    join_opts.lambda = nan;
    join_opts.window = 'not-set';
    join_opts.window_lr = 0;
    join_opts.bias = false;
    join_opts.adjust = true;
    join_opts.sigma = 0;
    join_opts.target_lr = 0;
    
    join_opts.adjustGainInit = 1;
    join_opts.adjustBiasInit = 0;
    % Learning rates ignored if batch-norm is enabled.
    join_opts.adjustGainLR = 0;
    join_opts.adjustBiasLR = 1;
    
    join_opts = vl_argparse(join_opts, {opts.join.conf});
    convOpts = {'CudnnWorkspaceLimit', 1024*1024*1024};

    join = dagnn.DagNN();
    % Apply window before correlation filter.
    join.addLayer('cf_window', MulConst(), ...
                  {'in1'}, {'cf_example'}, {'window'});
    p = join.getParamIndex('window');
    join.params(p).value = single(make_window(in_sz, join_opts.window));
    join.params(p).learningRate = join_opts.window_lr;

    % Establish whether there is a bias parameter to the XCorr.
    cf_outputs = {'tmpl'};
    xcorr_inputs = {'tmpl_cropped', 'in2'};

    % learnt alphas instead of CF for Adaptation Experiment
    if join_opts.const_cf
        join.addLayer('circ', ConvCircScalar(), ...
                      {'cf_example'}, cf_outputs, {'circf'});
        p = join.getParamIndex('circf');
        join.params(p).value = init_weight(opts.init, in_sz(1), in_sz(2), 1, 1, 'single');
    else
    % Add a correlation filter before the XCorr in branch 1.
        if join_opts.bias
            % Connect correlation filter bias to xcorr bias.
            cf_outputs = [cf_outputs, {'bias'}];
            xcorr_inputs = [xcorr_inputs, {'bias'}];
        end
        join.addLayer('cf', ...
                      CorrFilter('lambda', join_opts.lambda, ...
                                 'bias', join_opts.bias), ...
                      {'cf_example'}, cf_outputs, {'cf_target'});
        % Set correlation filter target.
        p = join.getParamIndex('cf_target');
    end
    
    join.addLayer('crop_z', ...
                    CropMargin('margin', 16), ...
                    cf_outputs, xcorr_inputs{1});
                
    % Cross-correlate template with features of other image.
    join.addLayer('xcorr', XCorr('bias', join_opts.bias), ...
                  xcorr_inputs, {'out'});


    assert(join_opts.sigma > 0);
    join.params(p).value = single(gaussian_response(in_sz, join_opts.sigma/in_stride));
    join.params(p).learningRate = join_opts.target_lr;

    % Add scalar layer to calibrate corr-filt scores for loss function.
    final.layers = {};
    if join_opts.adjust
        if  join_opts.finalBatchNorm
            % Batch-norm layer only.
            final.layers{end+1} = struct(...
                'type', 'bnorm', 'name', 'adjust_bn', ...
                'weights', {{single(join_opts.adjustGainInit), ...
                             single(join_opts.adjustBiasInit), ...
                             zeros(1, 2, 'single')}}, ...
                'learningRate', [2 1 0.3], ...
                'weightDecay', [0 0]);
        else
            final.layers{end+1} = struct(...
                'type', 'conv', 'name', 'adjust', ...
                'weights', {{single(1), single(-0.5)}}, ...
                'learningRate', [1, 2], ...
                'weightDecay', [0 0], ...
                'opts', {convOpts});            
        end        
    end
end

function num_out = output_dim(opts)
    % TODO: Restructure options so that this code does not need to know about
    % different types of loss function?
    switch opts.loss.type
        case {'simple', 'structured'}
            num_out = 1;
        case 'regression'
            assert(opts.instanceSize==opts.exemplarSize, 'Exemplar and Instance should have the same size.');
            assert(opts.negatives==0 && opts.hardNegatives==0, 'No negative pairs for the moment.');
            num_out = 4;
        otherwise
            error('unknown loss');
    end
end

function [out_sz, out_stride] = output_size(net, in_sz)
    % Assume that net has 1 input and 1 output.
    if isa(net, 'dagnn.DagNN')
        input = only(net.getInputs());
        output = only(net.getOutputs());
        sizes = net.getVarSizes({input, in_sz});
        out_sz = sizes{net.getVarIndex(output)}(1:3);
        rfs = net.getVarReceptiveFields(input);
        out_stride = rfs(net.getVarIndex(output)).stride;
    else
        info = vl_simplenn_display(net, 'inputSize', in_sz);
        out_sz = info.dataSize(1:3, end);
        out_stride = info.receptiveFieldStride(:, end);
    end
    out_sz = reshape(out_sz, 1, []);
    assert(all(out_stride == out_stride(1)));
    out_stride = out_stride(1);
end
