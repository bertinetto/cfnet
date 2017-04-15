function [pyramid, pad_masks_x] = make_scale_pyramid(im, targetPosition, in_side_scaled, out_side, avgChans, stats, p)
    n = numel(in_side_scaled);
    in_side_scaled = round(in_side_scaled);
    pyramid = zeros(out_side, out_side, 3, n, 'single');
    pad_masks_x = false(out_side, out_side, 1, n, 'logical');
    if ~isempty(p.gpus)
        pyramid = gpuArray(pyramid);
    end
    max_target_side = in_side_scaled(end);
    min_target_side = in_side_scaled(1);
    search_side = round(out_side * max_target_side / min_target_side);
    [search_region, ~] = get_subwindow_tracking(im, targetPosition, [search_side search_side], [max_target_side max_target_side], avgChans, p.gpus);
    
	if p.subMean
        search_region = bsxfun(@minus, search_region, reshape(stats.x.rgbMean, [1 1 3]));
    end
    
    for s = 1:n
        target_side = round(out_side * in_side_scaled(s) / min_target_side);
        pyramid(:,:,:,s) = get_subwindow_tracking(search_region, (1+search_side*[1 1])/2, [out_side out_side], target_side*[1 1], avgChans, p.gpus);
    end
end
