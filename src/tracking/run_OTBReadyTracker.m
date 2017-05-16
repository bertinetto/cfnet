function results = run_OTBReadyTracker(seq, res_path, bSaveImage)
 % rename OTBReadyTracker to the name of the tracker you are using in the OTB evaluation
    startup;
    paths = env_paths_tracking();
    tracker_params.visualization = false;
    tracker_params.gpus = 1;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    tracker_params.net = 'some-pretrained-network';
    tracker_params.net_gray = 'some-pretrained-network_gray';
    tracker_params.visualization = false;
    tracker_params.join.method = 'corrfilt'; % or 'xcorr'
    root_folder = 'your/path/to/otb/benchmark/root/';
    % % hyperparameters that work well for the tracker version you are using
    % tracker_params.scaleStep = 
    % tracker_params.scalePenalty = 
    % tracker_params.scaleLR = 
    % tracker_params.wInfluence = 
    % tracker_params.zLR =  
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tracker_params.imgFiles = vl_imreadjpeg(strcat(root_folder,seq.s_frames),'numThreads', 12);
    [cx, cy, w, h] = get_axis_aligned_BB(seq.init_rect);
    tracker_params.targetPosition = [cy cx];
    tracker_params.targetSize = round([h w]);
    % Call the main tracking function
    [bboxes, ~] = tracker(tracker_params);
    results = struct();
    results.res = bboxes;
    results.type = 'rect';
end
