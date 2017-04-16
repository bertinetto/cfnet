function y = gaussian_response(rect_size, sigma)
%GAUSSIAN_RESPONSE create the (fixed) target response of the correlation filter response
    [i1, i2] = circ_grid(rect_size(1), rect_size(2));
    y = exp(-(i1.^2 + i2.^2) / (2 * sigma^2));
end
