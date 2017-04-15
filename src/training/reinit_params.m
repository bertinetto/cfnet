function reinit_params(net, varargin)
% REINIT_PARAMS re-initializes the parameters of a DAG.

opts.scale = 1;
opts.weightInitMethod = 'xavierimproved';
opts.initBias = 0.1;
opts = vl_argparse(opts, varargin);

for l = 1:numel(net.layers)
    if isempty(net.layers(l).params)
        continue;
    end
    if isa(net.layers(l).block, 'dagnn.Conv')
        sz = net.layers(l).block.size;
        filters_param = net.layers(l).params{1};
        net.params(net.getParamIndex(filters_param)).value = ...
            init_weight(opts, sz(1), sz(2), sz(3), sz(4), 'single');
        % Optional bias.
        if numel(net.layers(l).params) > 1
            bias_param = net.layers(l).params{2};
            net.params(net.getParamIndex(bias_param)).value = ...
                ones(sz(4), 1, 'single') * opts.initBias;
        end
    elseif isa(net.layers(l).block, 'dagnn.BatchNorm')
        gain_param    = net.layers(l).params{1};
        bias_param    = net.layers(l).params{2};
        moments_param = net.layers(l).params{3};
        out = net.layers(l).block.numChannels;
        if isempty(out) || out < 1
            out = numel(net.params(net.getParamIndex(gain_param)).value);
            if out == 0
                error('output size unknown');
            end
        end
        net.params(net.getParamIndex(gain_param)).value = ...
            ones(out, 1, 'single');
        net.params(net.getParamIndex(bias_param)).value = ...
            zeros(out, 1, 'single');
        net.params(net.getParamIndex(moments_param)).value = ...
            zeros(out, 2, 'single');
    else
        % Every layer with params should be reinitialized.
        error(['unknown layer with params: ' class(net.layers(l).block)]);
    end
end

end

function weights = init_weight(opts, h, w, in, out, type)
% From examples/imagenet/cnn_imagenet_init.m

    % See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
    % rectifiers: Surpassing human-level performance on imagenet
    % classification. CoRR, (arXiv:1502.01852v1), 2015.
    switch lower(opts.weightInitMethod)
      case 'gaussian'
        sc = 0.01/opts.scale ;
        weights = randn(h, w, in, out, type)*sc;
      case 'xavier'
        sc = sqrt(3/(h*w*in)) ;
        weights = (rand(h, w, in, out, type)*2 - 1)*sc ;
      case 'xavierimproved'
        sc = sqrt(2/(h*w*out)) ;
        weights = randn(h, w, in, out, type)*sc ;
      otherwise
        error('Unknown weight initialization method''%s''', opts.weightInitMethod) ;
    end
end
