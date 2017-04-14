function net_dst = copy_network_params(net_src_path, net_dst_path)
% works only if params in the two networks have matching names
    net_src = load(net_src_path, 'net');
    net_dst = load(net_dst_path, 'net');
    % get cell array of params
    params_src = {net_src.net.params(:).name};
    params_dst = {net_dst.net.params(:).name};
    [to_copy, where_to_copy] = ismember(params_src, params_dst);
    % copy all the params with the same name
    for i = 1:numel(to_copy)
       if to_copy(i)
           fprintf('Copying %s ...\n', net_dst.net.params(i).name);
           assert(strcmp(net_dst.net.params(i).name, net_src.net.params(where_to_copy(i)).name));
           net_dst.net.params(i).value = net_src.net.params(where_to_copy(i)).value;
           net_dst.net.params(i).learningRate = net_src.net.params(where_to_copy(i)).learningRate;
           net_dst.net.params(i).weightDecay = net_src.net.params(where_to_copy(i)).weightDecay;
       end
    end
end