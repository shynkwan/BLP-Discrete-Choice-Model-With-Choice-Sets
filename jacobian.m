function Ddelta = jacobian(x0)

% JACOBIAN
% Computes the Jacobian of the mean utilities associated with
% the random coefficients Logit demand model.
%
% source: Dube, Fox and Su (2009)
% Code Revised: March 2009

global prods T nn v K x delta IV d pAll pSub oo xindex

dSdtheta2 = zeros(T*prods,K+2);
dSddeltaDIAG = zeros(T*prods,prods);
dSddelta = zeros(T*prods, T*prods);

theta1 = x0(1:K+1, 1);                      % mean tastes
theta2 = x0(K+2:2*K+2, 1);                  % st. deviation of tastes
Pi = x0(2*K+3,1);                            % income coefficient
delta = x0(2*K+4:2*K+3+T*prods, 1);         % mean utilities
g = x0(2*K+4+T*prods:end, 1);               % moment condition values

% Indices
nx0 = size(x0,1);
ng = size(g,1);
ooo = ones(1,K+1);
ooo1 = ones(prods,1);
 

cong = g - IV'*(delta - x*theta1);  % constraints on moment conditions

expmu = exp(x*diag(theta2)*v +  kron(x(:,1), oo) .* kron(d, ooo1 )*Pi);      % exponentiated deviations from mean utilities
expmeanval = exp(delta);

[EstShare, simShare] = ind_shnormMPEC(expmeanval,expmu);
[subEstShare, subSimShare] = sub_ind_shnormMPEC(expmeanval,expmu);

for tt=1:T,
    index = (0:prods-1)'*T+tt;
    for rr = 1:nn,
        dSddeltaDIAG(index,:) = dSddeltaDIAG(index, :) + pAll(tt) * (diag(simShare(index,rr)) - simShare(index,rr)*simShare(index,rr)')/nn ...
                            +  pSub(tt) *(diag(subSimShare(index,rr)) - subSimShare(index,rr)*subSimShare(index,rr)')/nn ;
        dSdtheta2(index,1:K+1) = dSdtheta2(index,1:K+1) + pAll(tt) * (simShare(index,rr)*ooo).*(ooo1*v(:,rr)').*( x(index,:) - (ooo1*(simShare(index,rr)'*x(index,:))))/nn ...
                                + pSub(tt) * (subSimShare(index,rr)*ooo).*(ooo1*v(:,rr)').*( x(index,:) - (ooo1*(subSimShare(index,rr)'*x(index,:))))/nn  ;
        dSdtheta2(index,K + 2) = dSdtheta2(index,K + 2) + pAll(tt) *  (simShare(index,rr)).*(ooo1 * d(tt,rr)).*( x(index,xindex) - (ooo1*(simShare(index,rr)'*x(index,xindex))))/nn ...
                             + pSub(tt) * (subSimShare(index,rr)).*(ooo1 * d(tt,rr)).*( x(index,xindex) - (ooo1*(subSimShare(index,rr)'*x(index,xindex))))/nn ;
    end
end

dSddelta = zeros(T*prods, T*prods);

for i = 1:prods
    for j = 1:prods
    dSddelta( (j-1)*T+1:j*T, (i-1)*T+1:i*T ) = diag(dSddeltaDIAG((j-1)*T+1:j*T, i)); 
    end
end

Ddelta = -inv(dSddelta)*dSdtheta2;
