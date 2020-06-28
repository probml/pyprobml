% based on tip at http://sachinashanbhag.blogspot.com/2012/09/setting-up-random-number-generator-seed.html
% forcing old generators
function rng(x)
  randn("seed",x)
  rand("seed",x)
end