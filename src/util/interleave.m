function s = interleave(a, b)
	n = numel(a);
	assert(numel(a) == numel(b));

	s = cell(1, 2*n);
	s(1:2:end) = a;
	s(2:2:end) = b;
end
