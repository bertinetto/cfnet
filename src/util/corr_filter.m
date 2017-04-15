function varargout = corr_filter(x, y, der_w, varargin)

opts.lambda = nan;
opts = vl_argparse(opts, varargin);

% x is [m1, m2, p, b]
% y is [m1, m2]
% der_w is same size as x

sz = size_min_ndims(x, 4);
n = prod(sz(1:2));

y_f = fft2(y);
x_f = fft2(x);
% k = 1/n sum_i x_i corr x_i + lambda delta
assert(~isnan(opts.lambda), 'lambda must be specified');
k_f = 1/n*sum(conj(x_f).*x_f, 3) + opts.lambda;
% a must satisfy n (k conv a) = y
% The signal a contains a weight per example (shift)
a_f = 1/n*bsxfun(@times, y_f, 1./k_f);

if isempty(der_w)
    % Use same weight a for all channels i.
    % w[i] = a corr x[i]
    w_f = bsxfun(@times, conj(a_f), x_f);
    % w = ifft2(w_f, 'symmetric');
    w = real(ifft2(w_f));
    varargout = {w};
else
    der_w_f = fft2(der_w);
    % a, x -> w
    % w[i] = a corr x[i]
    % dw[i] = da corr x[i] + a corr dx[i]
    % F dw[i] = conj(F da) .* F x[i] + conj(F a) .* F dx[i]
    % <der_w, dw> = sum_i <der_w[i], dw[i]> = sum_i <F der_w[i], F dw[i]>
    %   = sum_i <F der_w[i], conj(F da) .* F x[i] + conj(F a) .* F dx[i]>
    %   = <F da, sum_i conj(F der_w[i]) .* F x[i]> + sum_i <F der_w[i] .* F a, F dx[i]>
    der_a_f = sum(x_f .* conj(der_w_f), 3);
    der_x_f = bsxfun(@times, a_f, der_w_f);
    % k, y -> a
    % k conv a = 1/n y
    % dk conv a + k conv da = 1/n dy
    % dk_f .* a_f + k_f .* da_f = 1/n dy_f
    % <der_a, da> = <der_a_f, da_f>
    %   = <der_a_f, k_f^-1 .* (1/n dy_f - dk_f .* a_f)>
    %   = <1/n der_a_f .* conj(k_f^-1), dy_f> + <-der_a_f .* conj(k_f^-1 .* a_f), dk_f>
    %   = <der_y_f, dy_f> + <der_k_f, dk_f>
    der_y_f = 1/n*sum(der_a_f .* conj(1 ./ k_f), 4); % accumulate gradients over batch
    der_y = real(ifft2(der_y_f));
    der_k_f = -der_a_f .* conj(a_f ./ k_f);
    % x -> k
    % k = 1/n sum_i x_i corr x_i + lambda delta
    % dk = 1/n sum_i {dx[i] corr x[i] + x[i] corr dx[i]}
    % F dk = 1/n sum_i {conj(F dx[i]) .* F x[i] + conj(F x[i]) .* F dx[i]}
    % <der_k, dk> = <der_k, 1/n sum_i {dx[i] corr x[i] + x[i] corr dx[i]}>
    %   = sum_i <F der_k, 1/n conj(F dx[i]) .* F x[i] + conj(F x[i]) .* F dx[i]>
    %   = sum_i <F dx[i], 1/n conj(F der_k) .* F x[i]> + <1/n F der_k .* F x[i], F dx[i]>
    %   = sum_i <F dx[i], 1/n [F der_k + conj(F der_k)] .* F x[i]>
    %   = sum_i <F dx[i], 2/n real(F der_k) .* F x[i]>
    %   = sum_i <F der_x[i], F dx[i]>
    der_x_f = der_x_f + 2/n*bsxfun(@times, real(der_k_f), x_f);
    % der_x = ifft2(der_x_f, 'symmetric');
    der_x = real(ifft2(der_x_f));
    varargout = {der_x, der_y};
end

end
