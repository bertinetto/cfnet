function e = iou_error(x, label, ref_sz, pred_sz, stride)
% The score map x is [m1, m2, 1, b].
% numel(label) is b
% The dimensions m1 and m2 are odd numbers.
% ref_sz is [2, b]
% pred_sz is [2, b]
% stride is an integer

[m1, m2, k, b] = size(x);
assert(mod(m1, 2) == 1);
assert(mod(m2, 2) == 1);
assert(k == 1);
assert(stride >= 1);

pos = (label > 0);
num_pos = nnz(pos);
x = x(:,:,:,pos);
ref_sz = ref_sz(:,pos);
pred_sz = pred_sz(:,pos);

% Compute IoU.
% Range of displacements is -r1:r1 and -r2:r2.
% 2*r + 1 = m
r1 = (m1-1) / 2;
r2 = (m2-1) / 2;
d1 = stride * (-r1:r1);
d2 = stride * (-r2:r2);

[u1, u2] = find_max(x);

d = [d1(u1); d2(u2)];
[ds1,ds2,ds3] = size(d);
if ds2==1
    d = reshape(d, [ds1 ds3]);
end
a = max(-ref_sz/2, -pred_sz/2 + d);
b = min(ref_sz/2, pred_sz/2 + d);
i = prod(max(b-a, 0), 1);
u = prod(pred_sz, 1) + prod(ref_sz, 1) - i;
iou = i ./ u;
e = sum(iou);

end

function [u1, u2] = find_max(x)
    sz = size_min_ndims(x, 3);
    x = reshape(x, [sz(1)*sz(2), sz(3:end)]);
    [~, ind] = max(x);
    [u1, u2] = ind2sub(sz(1:2), ind);
end
