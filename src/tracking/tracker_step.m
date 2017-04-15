
function [newTargetPosition, bestScale] = tracker_step(net_x, s_x, s_z, scoreId, z_out, x_crops, pad_masks_x, targetPosition, window, p)
    % run a forward pass of the CNN
    net_x.eval([z_out, {'instance', x_crops}]);
    responseMaps = reshape(net_x.vars(scoreId).value, [p.scoreSize p.scoreSize p.numScale]);
    % init upsampled response map
    responseMapsUP = gpuArray(single(zeros(p.scoreSize*p.responseUp, p.scoreSize*p.responseUp, p.numScale)));
    % get the response map
    currentScaleID = ceil(p.numScale/2);
    % pick the response with the highest peak ratio
    bestScale = currentScaleID;
    bestPeak = -Inf;
    for s=1:p.numScale
        responseMapsUP(:,:,s) = imresize(responseMaps(:,:,s), p.responseUp, 'bicubic');
        thisResponse = responseMapsUP(:,:,s);
        % penalize change of scale
        if s~=currentScaleID
            thisResponse = thisResponse * p.scalePenalty;
        end
        thisPeak = max(thisResponse(:));
        if thisPeak > bestPeak
            bestPeak = thisPeak;
            bestScale = s;
        end
    end

    responseMap = responseMapsUP(:,:,bestScale);
    responseMap = responseMap - min(responseMap(:));
    
    % apply displacement-penalty window
    responseMap = responseMap / sum(responseMap(:));
    response_final = (1-p.wInfluence)*responseMap + p.wInfluence*window;
    %% update position and scale
    [r_max, c_max] = find(response_final == max(response_final(:)), 1);
    [r_max, c_max] = avoid_empty_position(r_max, c_max, p);
    p_corr = [r_max, c_max];
    % displacement from the center in instance final representation ...
    disp_instanceFinal = p_corr - (p.scoreSize*p.responseUp + 1)/2;
    % ... in instance input ...
    disp_instanceInput = disp_instanceFinal * p.totalStride / p.responseUp;
    % ... in instance original crop (in frame coordinates)
    disp_instanceFrame = disp_instanceInput * s_x / p.instanceSize;
    % position within frame in frame coordinates
    newTargetPosition = targetPosition + disp_instanceFrame;
end

function [r_max, c_max] = avoid_empty_position(r_max, c_max, params)
    if isempty(r_max)
        r_max = ceil(params.scoreSize/2);
    end
    if isempty(c_max)
        c_max = ceil(params.scoreSize/2);
    end
end
