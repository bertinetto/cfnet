function net = init_net(net, gpu, reset_gpu)
    % remove loss layer
    net = remove_layers_from_block(net, 'dagnn.Loss');
    % init specified GPU
    % init specified GPU
    if ~isempty(gpu)
        if reset_gpu
            gpuDevice(gpu);
        end
        net.move('gpu');
    end
    net.mode = 'test'; % very important for batch norm
end