function found = find_layers_from_prefix(net, prefix)
    found = false;
    L = net.layers;
    num_layers = numel(L);
    for i = 1:num_layers
        if strfind(L(i).name, prefix)
            found = true;
            break
        end
    end        
end