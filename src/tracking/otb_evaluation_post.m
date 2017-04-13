function otb_evaluation_post(epochs, expm_folders, expm_id, join_method, gpus)
% Process otb evaluation for already-trained network at EPOCHS
% Pass two string as cell array in expm_folders to copy params of src on dst.

        cache_file = [expm_folders{end} '/data/otb-eval_' expm_id '.mat'];
        if exist(cache_file, 'file')
            cache = load(cache_file);
            % remove incomplete results and epochs
            cache = remove_incomplete(cache);
        else
            cache = struct();
            cache.epochs = [];
        end
        % merge cached and new epochs
        epochs = unique([cache.epochs epochs]);
        ne = numel(epochs);
        curve_d = cell(1,ne);
        curve_o = cell(1,ne);
        TREd = zeros(1, ne);
        TREo = zeros(1, ne);
        % fill cached results in new data matrix
        cached_ids = find(ismember(epochs, cache.epochs)==1);
        fprintf('Found cached results for epochs: [%s]\n', int2str(epochs(cached_ids)));
        for i=1:numel(cached_ids)
            curve_d{cached_ids(i)} = cache.curve_d{i};
            curve_o{cached_ids(i)} = cache.curve_o{i};
            TREd(cached_ids(i)) = cache.TREd(i);
            TREo(cached_ids(i)) = cache.TREo(i);
        end

        tracker_params = {};
        tracker_params.gpus = gpus;
        tracker_params.join.method = join_method;
        tracker_params.paths.net_base = '';

        new_e = find(ismember(epochs, cache.epochs)==0);
        fprintf('Running otb-evaluation-post for epochs: [%s] ...\n', int2str(epochs(new_e)));
        for i=1:numel(new_e)
            fprintf('\tEpoch %d\n', epochs(new_e(i)));
            if i>1, tracker_params.init_gpu = false; end
            if numel(expm_folders)==1
                tracker_params.net = [expm_folders{1} '/data/net-epoch-' int2str(epochs(new_e(i))) '.mat'];
            else
                tracker_params.net = copy_network_params([expm_folders{1} '/data/net-epoch-' int2str(epochs(new_e(i))) '.mat'], [expm_folders{2} '/data/net-epoch-' int2str(epochs(new_e(i))) '.mat']);
            end
            [curve_d{new_e(i)}, curve_o{new_e(i)}, TREd(new_e(i)), TREo(new_e(i)), ~, ~, ~, ~] = run_tracker_OTB_TRE('all', tracker_params);
            % remove empty results just for printing
            [epochs_, curve_d_, curve_o_, TREd_, TREo_] = remove_incomplete_print(epochs, curve_d, curve_o, TREd, TREo);
            print_OTB(epochs_, TREd_, TREo_, expm_folders{end}, expm_id);
            save(cache_file, 'epochs', 'curve_d', 'curve_o', 'TREd', 'TREo');
        end
        fprintf('\n\nDone.');
end

function cache = remove_incomplete(cache_in)

        cache = cache_in;
        incomplete = find(cache.TREd==0);
        cache.epochs(incomplete) = [];
        cache.TREd(incomplete) = [];
        cache.TREo(incomplete) = [];
end

function [epochs, curve_d, curve_o, TREd, TREo] = remove_incomplete_print(epochs, curve_d, curve_o, TREd, TREo)

        incomplete = find(TREd==0);
        epochs(incomplete) = [];
        curve_d(incomplete) = [];
        curve_o(incomplete) = [];
        TREd(incomplete) = [];
        TREo(incomplete) = [];
end

function print_OTB(epochs, dist, overlap, expm_folder, expm_id)

    figure(1), subplot(1,2,1)
    plot(epochs, dist, 'r+-');
    xlabel('Epoch'); ylabel('Center distance'); grid on; grid minor
    drawnow;

    subplot(1,2,2)
    plot(epochs, overlap, 'b+-');
    xlabel('Epoch'); ylabel('Overlap'); grid on; grid minor
    drawnow;

    print(1, [expm_folder '/data/OTB-dist_' expm_id '.pdf'], '-dpdf') ;

end
