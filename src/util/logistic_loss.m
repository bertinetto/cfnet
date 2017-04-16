function y = logistic_loss(x,c,instanceWeights,dzdy,varargin)
%VL_NNLOSS CNN categorical or attribute loss.
%   Y = VL_NNLOSS(X, C) computes the loss incurred by the prediction
%   scores X given the categorical labels C.
%
%   The prediction scores X are organised as a field of prediction
%   vectors, represented by a H x W x D x N array. The first two
%   dimensions, H and W, are spatial and correspond to the height and
%   width of the field; the third dimension D is the number of
%   categories or classes; finally, the dimension N is the number of
%   data items (images) packed in the array.
%
%   While often one has H = W = 1, the case W, H > 1 is useful in
%   dense labelling problems such as image segmentation. In the latter
%   case, the loss is summed across pixels (contributions can be
%   weighed using the `InstanceWeights` option described below).
%
%   The array C contains the categorical labels. In the simplest case,
%   C is an array of integers in the range [1, D] with N elements
%   specifying one label for each of the N images. If H, W > 1, the
%   same label is implicitly applied to all spatial locations.
%
%   In the second form, C has dimension H x W x 1 x N and specifies a
%   categorical label for each spatial location.
%
%   In the third form, C has dimension H x W x D x N and specifies
%   attributes rather than categories. Here elements in C are either
%   +1 or -1 and C, where +1 denotes that an attribute is present and
%   -1 that it is not. The key difference is that multiple attributes
%   can be active at the same time, while categories are mutually
%   exclusive. By default, the loss is *summed* across attributes
%   (unless otherwise specified using the `InstanceWeights` option
%   described below).
%
%   DZDX = VL_NNLOSS(X, C, DZDY) computes the derivative of the block
%   projected onto the output derivative DZDY. DZDX and DZDY have the
%   same dimensions as X and Y respectively.
%
%   Logistic log loss:: `logisticlog`
%     L(x,c) = log(1 + exp(- cx)). This is the same as the `binarylog`
%     loss, but implicitly normalizes the score x into a probability
%     using the logistic (sigmoid) function: p = sigmoid(x) = 1 / (1 +
%     exp(-x)). This is also equivalent to `softmaxlog` loss where
%     class c=+1 is assigned score x and class c=-1 is assigned score
%     0.
%
%
%   VL_NNLOSS(...,'OPT', VALUE, ...) supports these additionals
%   options:
%
%   InstanceWeights:: []
%     Allows to weight the loss as L'(x,c) = WGT L(x,c), where WGT is
%     a per-instance weight extracted from the array
%     `InstanceWeights`. For categorical losses, this is either a H x
%     W x 1 or a H x W x 1 x N array. For attribute losses, this is
%     either a H x W x D or a H x W x D x N array.
%
%   TopK:: 5
%     Top-K value for the top-K error. Note that K should not
%     exceed the number of labels.
%
%   See also: VL_NNSOFTMAX().

% Copyright (C) 2014-15 Andrea Vedaldi.
% Copyright (C) 2016 Karel Lenc.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.classWeights = [] ;
opts.threshold = 0 ;
opts.loss = 'softmaxlog' ;
opts.topK = 5 ;
opts = vl_argparse(opts, varargin, 'nonrecursive') ;

inputSize = [size(x,1) size(x,2) size(x,3) size(x,4)] ;

% Form 1: C has one label per image. In this case, get C in form 2 or
% form 3.
c = gather(c) ;
if numel(c) == inputSize(4)
  c = reshape(c, [1 1 1 inputSize(4)]) ;
  c = repmat(c, inputSize(1:2)) ;
end

hasIgnoreLabel = any(c(:) == 0);

% --------------------------------------------------------------------
% Spatial weighting
% --------------------------------------------------------------------

% work around a bug in MATLAB, where native cast() would slow
% progressively
if isa(x, 'gpuArray')
  switch classUnderlying(x) ;
    case 'single', cast = @(z) single(z) ;
    case 'double', cast = @(z) double(z) ;
  end
else
  switch class(x)
    case 'single', cast = @(z) single(z) ;
    case 'double', cast = @(z) double(z) ;
  end
end

labelSize = [size(c,1) size(c,2) size(c,3) size(c,4)] ;
assert(isequal(labelSize(1:2), inputSize(1:2))) ;
assert(labelSize(4) == inputSize(4)) ;


% there must be one categorical label per prediction scalar
assert(labelSize(3) == inputSize(3)) ;

if hasIgnoreLabel
  % null labels denote instances that should be skipped
  instanceWeights = cast(c ~= 0) ;
end


% --------------------------------------------------------------------
% Do the work
% --------------------------------------------------------------------

if nargin <= 2 || isempty(dzdy)
    a = -c.*x ;
    b = max(0, a) ;
    t = b + log(exp(-b) + exp(a-b)) ;
    if ~isempty(instanceWeights)
        y = instanceWeights(:)' * t(:) ;
    else
        y = sum(t(:));
    end
else
    if ~isempty(instanceWeights)
        dzdy = dzdy * instanceWeights ;
    end

    y = - dzdy .* c ./ (1 + exp(c.*x)) ;

end

% --------------------------------------------------------------------
function y = onesLike(x)
% --------------------------------------------------------------------
if isa(x,'gpuArray')
  y = gpuArray.ones(size(x),classUnderlying(x)) ;
else
  y = ones(size(x),'like',x) ;
end
