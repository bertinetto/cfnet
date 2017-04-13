function d = dist_with_GT(estimate, ground_truth)
    % strip off NaNs used to maintain all groundtruth in same matrix
    ground_truth = ground_truth(~isnan(ground_truth));
    if numel(ground_truth)==4
        cxgt = ground_truth(1)+ground_truth(3)/2;
        cygt = ground_truth(2)+ground_truth(4)/2;
    else
        [cxgt, cygt, ~, ~] = get_axis_aligned_BB(ground_truth);
    end
    cxe = estimate(1)+estimate(3)/2;
    cye = estimate(2)+estimate(4)/2;
    d = sqrt((cxgt - cxe).^2 + (cygt - cye).^2);
    d = min(d, 50); % saturate
end
