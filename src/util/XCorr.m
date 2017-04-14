% Code for the method described in the paper
% "Fully-Convolutional Siamese Networks for Object Tracking", L. Bertinetto, J. Valmadre, J. Henriques, A. Vedaldi, P. Torr.
% Project page: http://www.robots.ox.ac.uk/~luca/siamese-fc.html
%
% Copyright (C) 2016 Luca Bertinetto, Joao Henriques and Jack Valmadre.
% All rights reserved.


classdef XCorr < dagnn.Layer

    properties
        bias = false;
        opts = {'cuDNN'}
    end

    methods
        function outputs = forward(obj, inputs, params)
            if obj.bias
                assert(numel(inputs) == 3, 'three inputs are needed');
            else
                assert(numel(inputs) == 2, 'two inputs are needed');
            end

            if obj.bias
                outputs{1} = cross_corr(inputs{1:3});
            else
                outputs{1} = cross_corr(inputs{1:2}, []);
            end
        end

        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            if obj.bias
                assert(numel(inputs) == 3, 'three inputs are needed');
            else
                assert(numel(inputs) == 2, 'two inputs are needed');
            end
            assert(numel(derOutputs) == 1, 'only one gradient should be flowing in this layer (dldy)');

            if obj.bias
                [derInputs{1:3}] = cross_corr(inputs{1:3}, derOutputs{1});
            else
                [derInputs{1:2}] = cross_corr(inputs{1:2}, [], derOutputs{1});
            end
            derParams = {};
        end

        function outputSizes = getOutputSizes(obj, inputSizes)
            z_sz = inputSizes{1};
            x_sz = inputSizes{2};
            y_sz = [x_sz(1:2) - z_sz(1:2) + 1, 1, z_sz(4)];
            outputSizes = {y_sz};
        end

        function rfs = getReceptiveFields(obj)
            rfs(1,1).size = [inf inf]; % could be anything
            rfs(1,1).stride = [1 1];
            rfs(1,1).offset = 1;
            rfs(2,1).size = [inf inf];
            rfs(2,1).stride = [1 1];
            rfs(2,1).offset = 1;
            rfs(3,1).size = [inf inf];
            rfs(3,1).stride = [1 1];
            rfs(3,1).offset = 1;
        end

        function obj = XCorr(varargin)
            obj.load(varargin);
        end

    end

end
