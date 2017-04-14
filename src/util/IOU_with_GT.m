function IOU = IOU_with_GT(estimate, ground_truth)
    % strip off NaNs used to maintain all groundtruth in same matrix
    ground_truth = ground_truth(~isnan(ground_truth));
    % axis-aligned rectangle format
    if numel(ground_truth)==4
        if sum(estimate)==0
            IOU = 0;
        else
            IOU = bboxOverlapRatio(estimate, ground_truth);
        end
    else
        % rotated bounding box format (VOT)
        warning off;
        e_x = [estimate(1) estimate(1)+estimate(3) estimate(1)+estimate(3) estimate(1)];
        e_y = [estimate(2) estimate(2) estimate(2)+estimate(4) estimate(2)+estimate(4)];
        gt_x = ground_truth(1:2:7);
        gt_y = ground_truth(2:2:8);
        [intersection_x, intersection_y] = polybool('intersection',e_x,e_y,gt_x,gt_y);
        [union_x, union_y] = polybool('union',e_x,e_y,gt_x,gt_y);

        interseaction_area = polyarea(intersection_x, intersection_y);
        union_area = polyarea(union_x, union_y);

        IOU = interseaction_area / union_area;
        if isnan(IOU)
            IOU = 0;
        end
    end
end
