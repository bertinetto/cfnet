function print_OTB(epochs, epochs_to_test, AUC_OPEd, AUC_TREd, AUC_OPEo, AUC_TREo, expm_folder, expm_id, zLRs)

    color_specs = {'r-+','g-+','b-+','c-+','m-+','y-+','k-+'};

    if size(AUC_OPEd,2) > numel(color_specs)
       error('Too many lines to plot')
    end

    handles = zeros(1, numel(zLRs));

    subplot(1,2,1)
    for i=1:numel(zLRs)
        handles(i) = plot(epochs, AUC_OPEd(:, i), color_specs{i}, 'DisplayName', sprintf('zLR=%.3f', zLRs(i)));
        hold on
    end
    hold off
    xlabel('Epoch');
    title('AUC of OTB-OPE (dist)')
    grid on
    grid minor
    legend(handles,'Location','best')
%     h = get(gca);
%     set(gca, 'YTick', linspace(h.YLim(1), h.YLim(2), 2*numel(h.YTick)-2));

    subplot(1,2,2)
    for i=1:numel(zLRs)
        handles(i) = plot(epochs, AUC_TREd(:, i), color_specs{i}, 'DisplayName', sprintf('zLR=%.3f', zLRs(i)));
        hold on
    end
    hold off
    xlabel('Epoch');
    title('AUC of OTB-TRE (dist)')
    grid on
    grid minor
    legend(handles,'Location','best')
%     h = get(gca);
%     set(gca, 'YTick', linspace(h.YLim(1), h.YLim(2), 2*numel(h.YTick)-2));

    drawnow ;
    print(1, [expm_folder '/data/OTB-dist_' expm_id '.pdf'], '-dpdf') ;
    subplot(1,2,1)
    for i=1:numel(zLRs)
        handles(i) = plot(epochs, AUC_OPEo(:, i), color_specs{i}, 'DisplayName', sprintf('zLR=%.3f', zLRs(i)));
        hold on
    end
    hold off
    xlabel('Epoch');
    title('AUC of OTB-OPE (IOU)')
    grid on
    grid minor
    legend(handles,'Location','best')
%     h = get(gca);
%     set(gca, 'YTick', linspace(h.YLim(1), h.YLim(2), 2*numel(h.YTick)-2));

    subplot(1,2,2)
    for i=1:numel(zLRs)
        handles(i) = plot(epochs, AUC_TREo(:, i), color_specs{i}, 'DisplayName', sprintf('zLR=%.3f', zLRs(i)));
        hold on
    end
    hold off
    xlabel('Epoch');
    title('AUC of OTB-TRE (IOU)')
    grid on
    grid minor
    legend(handles,'Location','best')
%     h = get(gca);
%     set(gca, 'YTick', linspace(h.YLim(1), h.YLim(2), 2*numel(h.YTick)-2));

    drawnow ;
    print(1, [expm_folder '/data/OTB-iou_' expm_id '.pdf'], '-dpdf') ;

end
