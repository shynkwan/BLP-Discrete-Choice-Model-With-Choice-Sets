function [f grad_f] = GMMMPEC_f_ktr(x0)

% GMMMPEC_f
% GMM Objective function for the random coefficients Logit esitmated via
% MPEC.



global W K prods T

theta1 = x0(1:K+1, 1);                      % mean tastes
theta2 = x0(K+2:2*K+2, 1);                  % st. deviation of tastes
Pi = x0(2*K+3,1);                            % income coefficient
delta = x0(2*K+4:2*K+3+T*prods, 1);         % mean utilities
g = x0(2*K+4+T*prods:end, 1);               % moment condition values

f = g'*W*g; 

if nargout==2,
    nx0 = size(x0,1);
    grad_f = zeros(nx0,1);
    grad_f(2*K+4+T*prods:end,1) = 2*W*g;
end