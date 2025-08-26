function distorted_data = distort_sample(data, a, m, seed)
% DISTORT_SAMPLE  MATLAB port of the provided R function.
% data: table containing at least variable 'fL'
% a: fraction of rows to distort (0..1)
% m: mean for replacement N(m, 0.1^2)
% seed: RNG seed

  distorted_data = data;
  n = height(data);
  num_to_distort = ceil(a * n);

  rng(seed);
  idx = sort(randperm(n, num_to_distort));
  distorted_data.fL(idx) = m + 0.1 * randn(num_to_distort, 1);
end
