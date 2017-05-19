%% Sample execution for CFNet-conv1
% hyper-parameters reported in Supp.material for CVPR'17, Table 2 for arXiv version
tracker_par.join.method = 'corrfilt';
tracker_par.net = 'cfnet-conv1_e75.mat';
tracker_par.net_gray = 'cfnet-conv1_gray_e55.mat';
tracker_par.scaleStep = 1.0355;
tracker_par.scalePenalty = 0.9825;
tracker_par.scaleLR = 0.7;
tracker_par.wInfluence = 0.2375;
tracker_par.zLR = 0.0058;

[~,~,dist,overlap,~,~,~,~] = run_tracker_evaluation('all', tracker_par);
