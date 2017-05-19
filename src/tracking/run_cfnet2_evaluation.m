%% Sample execution for CFNet-conv2
% hyper-parameters reported in Supp.material for CVPR'17, Table 2 for arXiv version
tracker_par.join.method = 'corrfilt';
tracker_par.net = 'cfnet-conv2_e80.mat';
tracker_par.net_gray = 'cfnet-conv2_gray_e40.mat';
tracker_par.scaleStep = 1.0575;
tracker_par.scalePenalty = 0.9780;
tracker_par.scaleLR = 0.52;
tracker_par.wInfluence = 0.2625;
tracker_par.zLR = 0.005;

[~,~,dist,overlap,~,~,~,~] = run_tracker_evaluation('all', tracker_par);
