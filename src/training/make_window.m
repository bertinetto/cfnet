function h = make_window(sz, type)
% sz = [m1, m2]

switch type
    case ''
        h = ones(sz(1:2));
    case 'cos'
        h = bsxfun(@times, reshape(hann(sz(1)), [sz(1), 1]), ...
                           reshape(hann(sz(2)), [1, sz(2)]));
    otherwise
        error(sprintf('unknown window: ''%s''', type));
end

end
