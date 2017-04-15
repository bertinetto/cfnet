function varargout = cross_corr(z, x, c, der_y)
% Computes y = z corr x + c

% z is [m1, m2, p, b]
% x is [n1, n2, p, b]
% c is [1, 1, 1, b] or empty
% der_y is [n1-m1+1, n2-m2+1, 1, b]

if nargin < 4
    der_y = [];
end

z_sz = size_min_ndims(z, 4);
x_sz = size_min_ndims(x, 4);
assert(all(z_sz(1:2) <= x_sz(1:2)), 'exemplar z has to be smaller than instance x');

r_sz = [x_sz(1:2) - z_sz(1:2) + 1, 1, x_sz(4)];
x_ = reshape(x, [x_sz(1:2), prod(x_sz(3:4)), 1]);

if isempty(der_y)
    % TODO: Not necessary to compute in backward pass.
    r_ = vl_nnconv(x_, z, []);
    assert(isequal(size_min_ndims(r_, 4), [r_sz(1:2), r_sz(4), 1]));
    r = reshape(r_, r_sz);
    if isempty(c)
        y = r;
    else
        y = bsxfun(@plus, r, c);
    end
    varargout = {y};
else
    % r, c -> y
    der_r = der_y;
    if ~isempty(c)
        der_c = sum(sum(der_y, 1), 2);
    end
    % x, z -> r
    der_r_ = reshape(der_r, [r_sz(1:2), r_sz(4), 1]);
    [der_x_, der_z] = vl_nnconv(x_, z, [], der_r_);
    der_x = reshape(der_x_, x_sz);
    if isempty(c)
        varargout = {der_z, der_x};
    else
        varargout = {der_z, der_x, der_c};
    end
end

end
