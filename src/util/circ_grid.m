function [i1, i2] = circ_grid(m1, m2)
    m = [m1, m2];
    [i1, i2] = ndgrid(0:m(1)-1, 0:m(2)-1);
    half = floor((m-1) / 2);
    i1 = mod(i1 + half(1), m(1)) - half(1);
    i2 = mod(i2 + half(2), m(2)) - half(2);
end
