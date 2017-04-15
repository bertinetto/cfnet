classdef IOUErrorScore < dagnn.Loss
  properties
    stride = 0
  end

  methods
    function outputs = forward(obj, inputs, params)
    % inputs{1} is a scalar response map
    % inputs{2} is a scalar label, -1 or 1
    % inputs{3} contains the size of the reference rectangle
    % inputs{4} contains the size of the predicted rectangle
        outputs{1} = iou_error(inputs{1}, inputs{2}, inputs{3}, inputs{4}, obj.stride);
        num_pos = nnz(inputs{2} > 0);
        n = obj.numAveraged;
        m = n + num_pos;
        obj.average = (n * obj.average + gather(outputs{1})) / m;
        obj.numAveraged = m;
    end

    function obj = IOUErrorScore(varargin)
      obj.load(varargin) ;
    end
  end
end
