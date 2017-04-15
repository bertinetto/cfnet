classdef CorrFilter < dagnn.Layer

    properties
        lambda = 1e-3; % regularization
        window = ''; % LEGACY
        bias = true;
    end

    methods
        function outputs = forward(obj, inputs, params)
            assert(numel(inputs) == 1, 'one input is needed');
            assert(numel(params) == 1, 'one param is needed');
            args = {'lambda', obj.lambda};
            if obj.bias
                [outputs{1}, outputs{2}] = corr_filter_bias(...
                    inputs{1}, params{1}, [], [], args{:});
            else
                outputs{1} = corr_filter(inputs{1}, params{1}, [], args{:});
            end
        end

        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            assert(numel(inputs) == 1, 'one input is needed');
            assert(numel(params) == 1, 'one param is needed');
            args = {'lambda', obj.lambda};
            if obj.bias
                assert(numel(derOutputs) == 2, 'expect two gradients');
                [derInputs{1}, derParams{1}] = corr_filter_bias(...
                    inputs{1}, params{1}, derOutputs{1}, derOutputs{2}, args{:});
            else
                assert(numel(derOutputs) == 1, 'expect one gradient');
                [derInputs{1}, derParams{1}] = corr_filter(...
                    inputs{1}, params{1}, derOutputs{1}, args{:});
            end
        end

        function outputSizes = getOutputSizes(obj, inputSizes)
            x_sz = inputSizes{1};
            w_sz = x_sz;
            b_sz = [1, 1, 1, x_sz(4)];
            if obj.bias
                outputSizes = {w_sz, b_sz};
            else
                outputSizes = {w_sz};
            end
        end

        function rfs = getReceptiveFields(obj)
            % x -> w
            rfs(1,1).size = [inf inf]; % fully connected
            rfs(1,1).stride = [1 1];
            rfs(1,1).offset = 1;
            if obj.bias
              % x -> b
              rfs(1,2).size = [inf inf];
              rfs(1,2).stride = [1 1];
              rfs(1,2).offset = 1;
            end
        end

        function obj = CorrFilter(varargin)
            obj.load(varargin);
        end
    end
end
