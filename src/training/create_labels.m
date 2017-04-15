function [posLabel, negLabel, posWeight, negWeight] = create_labels(posLabelSize, labelWeight, rPos, rNeg)
    assert(mod(posLabelSize(1),2)==1)
    half = floor(posLabelSize(1)/2)+1;
    switch labelWeight
        case 'balanced'
            % weight by class size (+/-)
            posLabel = create_logisticloss_label(posLabelSize, rPos, rNeg);
            negLabel = -1 * ones(posLabelSize(1));
            posWeight = ones(size(posLabel));
            sumP = numel(find(posLabel==1));
            sumN = numel(find(posLabel==-1));
            posWeight(posLabel==1) = 0.5 * posWeight(posLabel==1) / sumP;
            posWeight(posLabel==-1) = 0.5 * posWeight(posLabel==-1) / sumN;
            negWeight = 0.5 * ones(posLabelSize(1)) / numel(find(negLabel==-1));          
        otherwise
            error('Unknown option for instance weights');
    end
end
