classdef centerThrErr < dagnn.Loss
  properties
    stride = 0
  end
    methods
        function outputs = forward(obj, inputs, params)
            assert(obj.stride > 0);
            radiusInPixel = 50;
            nStep = 100;
            batch_size = size(inputs{1},4);
            pos_mask = inputs{2}(:) > 0;
            num_pos = sum(pos_mask);
            outputs{1} = 0;
            n = obj.numAveraged;
            % avg only on num pos, not entire batch
            m = n + num_pos;
            if numel(inputs)==2
                % fully-convolutional case
                half = floor(size(inputs{1},1)/2)+1;
                centerLabel = repmat([half half], [num_pos 1]);
                positions = zeros(num_pos, 2);
                responses = inputs{1};
                responses = responses(:,:,:,pos_mask);
                for b = 1:num_pos
                    score = gather(responses(:,:,1,b));
                    [r_max, c_max] = find(score == max(score(:)), 1);
                    positions(b, :) = [r_max c_max];
                end
                radius = radiusInPixel / obj.stride;
            else
                % regression
                bboxes = reshape(inputs{1}, [4 batch_size]);
                positions = bboxes(1:2,:)+bboxes(3:4,:)./2;
                centerLabel = repmat(127/2, [2 batch_size]);
                radius = radiusInPixel;
            end
            outputs{1} = precision_auc(positions, centerLabel, radius, nStep);
            obj.average = (n * obj.average + gather(outputs{1})) / m ;
            obj.numAveraged = m ;
        end

        function obj = centerThrErr(varargin)
          obj.load(varargin) ;
        end
  end
end
