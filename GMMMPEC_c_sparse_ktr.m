function [cineq, c, dcineq, dc] = GMMMPEC_c_sparse_ktr(x0) 

global x IV K prods T v  nn share ConsPattern d oo pAll pSub xindex % logshare

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
dc = ConsPattern'; 

cong = g - IV'*(delta - x*theta1);  % constraints on moment conditions

expmu = exp(x*diag(theta2)*v +  kron(x(:,xindex), oo) .* kron(d, ooo1 )*Pi);      % exponentiated deviations from mean utilities
expmeanval = exp(delta);

[EstShare, simShare] = ind_shnormMPEC(expmeanval,expmu);  % share conditional on observing all products
[subEstShare, subSimShare] = sub_ind_shnormMPEC(expmeanval,expmu);     % share conditional on observing a subset of the products

cineq = [];
dcineq = [];

c = [EstShare.*kron(pAll, ones(prods,1)) + subEstShare.*kron(pSub, ones(prods,1)) - share ;
     cong ]; 
     
    % 4) Evaluate the Gradients
    for tt=1:T,
        index = (1:prods)'+(tt-1)*prods;  
        dSddeltaDIAG = zeros(prods,prods);
        dSdtheta2_index = zeros(prods,K+1);
        dSdtheta3_index = zeros(prods,1);
        
        for rr = 1:nn,    
            dSddeltaDIAG = dSddeltaDIAG + pAll(tt) * (diag(simShare(index,rr)) - simShare(index,rr)*simShare(index,rr)')/nn ...
                            +  pSub(tt) *(diag(subSimShare(index,rr)) - subSimShare(index,rr)*subSimShare(index,rr)')/nn ;    
            dSdtheta2_index = dSdtheta2_index + pAll(tt) * (simShare(index,rr)*ooo).*(ooo1*v(:,rr)').*( x(index,:) - (ooo1*(simShare(index,rr)'*x(index,:))))/nn ...
                                + pSub(tt) * (subSimShare(index,rr)*ooo).*(ooo1*v(:,rr)').*( x(index,:) - (ooo1*(subSimShare(index,rr)'*x(index,:))))/nn  ;
            dSdtheta3_index = dSdtheta3_index + pAll(tt) *  (simShare(index,rr)).*(ooo1 * d(tt,rr)).*( x(index,xindex) - (ooo1*(simShare(index,rr)'*x(index,xindex))))/nn ...
                             + pSub(tt) * (subSimShare(index,rr)).*(ooo1 * d(tt,rr)).*( x(index,xindex) - (ooo1*(subSimShare(index,rr)'*x(index,xindex))))/nn ;
        end
        dc(K+2:2*K+2,index) = dSdtheta2_index';
        dc(2*K+3,index) = dSdtheta3_index';
        dc(2*K+3+index, index) = dSddeltaDIAG'; 
    end
   
    dc(:,T*prods+1:end) = [IV'*x, zeros(ng, K+2), -IV', speye(ng)]';
    
    
end