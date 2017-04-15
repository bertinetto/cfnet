classdef MulConst < dagnn.ElementWise
% MulConst multiplies its input by its parameter.

    properties
    end

    methods
        function outputs = forward(obj, inputs, params)
            assert(numel(inputs) == 1, 'one input is needed');
            assert(numel(params) == 1, 'one param is needed');
            outputs{1} = mul_const(inputs{1}, params{1});
        end

        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            assert(numel(inputs) == 1, 'one input is needed');
            assert(numel(params) == 1, 'one param is needed');
            assert(numel(derOutputs) == 1, 'expect one gradient');
            [derInputs{1}, derParams{1}] = mul_const(...
                inputs{1}, params{1}, derOutputs{1});
        end

        function obj = MulConst(varargin)
            obj.load(varargin);
        end
    end
end
