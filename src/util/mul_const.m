function varargout = mul_const(x, h, der_y)

% x is [m1, m2, p, b]
% h is [m1, m2]
% der_y is same size as x

if nargin < 3
    der_y = [];
end

if isempty(der_y)
    y = bsxfun(@times, x, h);
    varargout = {y};
else
    der_x = bsxfun(@times, der_y, h);
    der_h = sum(sum(der_y .* x, 3), 4);
    varargout = {der_x, der_h};
end

end
