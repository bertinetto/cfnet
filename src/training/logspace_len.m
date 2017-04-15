function s = logspace_len(a, b, n, l)

if nargin < 4
    l = n
end

r = exp((log(10^b)-log(10^a)) / (n-1));
s = 10^a * r.^(0:l-1);

end
