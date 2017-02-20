function [hess] = GMMMPEC_hess_sparse_ktr(x0,lambda) 

% GMMMPEC_c
% Constraints for the random coefficients Logit estimated via MPEC.
%
% source: Dube, Fox and Su (2009)
% Code Revised: March 2009


global x IV K prods T v nn logshare share W HessianPattern d oo pAll pSub xindex

theta1 = x0(1:K+1, 1);                      % mean tastes
theta2 = x0(K+2:2*K+2, 1);                  % st. deviation of tastes
Pi = x0(2*K+3,1);                            % income coefficient
delta = x0(2*K+4:2*K+3+T*prods, 1);         % mean utilities
g = x0(2*K+4+T*prods:end, 1);               % moment condition values

nx0 = size(x0,1);
ng = size(g,1);
ooo = ones(1,K+1);
ooo1 = ones(prods,1);

% hess = zeros(nx0, nx0);
hess = HessianPattern;
hess(2*K+4+T*prods:end, 2*K+4+T*prods:end) = 2*W;

expmu = exp(x*diag(theta2)*v +  kron(x(:,xindex), oo) .* kron(d, ooo1 )*Pi);      % exponentiated deviations from mean utilities
expmeanval = exp(delta);
[EstShare, simShare] = ind_shnormMPEC(expmeanval,expmu);
[subEstShare, subSimShare] = sub_ind_shnormMPEC(expmeanval,expmu);

dL2dtheta22 = zeros(K+1,K+1);
dL2dtheta22rr = zeros(K+1,K+1);

% Evaluate the hessian
for tt=1:T,    
    index = (1:prods)'+(tt-1)*prods; 
    dL2ddelta2DIAG = zeros(prods,prods);
    dL2ddeltadtheta_index = zeros(prods, K+1);
    dL2ddeltadpi_index = zeros(prods,1);
    dL2dpi2 = zeros(1,1);
    dL2dpidtheta2 = zeros(K + 1, 1);
    multip = lambda.eqnonlin(index);
% multip = ones(10,1);
    
    for rr = 1:nn,    
        
        %% d2L  of delta & delta
        
        % Full version
        simS = simShare(index,rr);    
        sumprod_mpsimS = multip'*simS;
            
        blk1 = sumprod_mpsimS*(-diag(simS) + 2*simS*simS');
        blk2 = -(multip.*simS)*simS';
        blk3 = diag(multip.*simS);
        blk = blk1 + blk2 + blk2'+ blk3;  
        
        % Subset version
        subSimS = subSimShare(index,rr);    
        subsumprod_mpsimS = multip'*subSimS;
            
        subblk1 = subsumprod_mpsimS*(-diag(subSimS) + 2*subSimS*subSimS');
        subblk2 = -(multip.*subSimS)*subSimS';
        subblk3 = diag(multip.*subSimS);
        subblk = subblk1 + subblk2 + subblk2'+ subblk3;    
        
        % Sum them      
        dL2ddelta2DIAG = dL2ddelta2DIAG +  pAll(tt)*blk/nn +  pSub(tt)*subblk/nn;
        
        %% d2L of delta & theta2
        
        % Full version
        xsimSx = x(index,:) - (ooo1*(simS'*x(index,:)));
        xsimSxv = xsimSx.*(ooo1*v(:,rr)'); 
        dSdtheta2rr = (simS*ooo).*xsimSxv;
           
        dL2ddeltadthetarr = -simS*multip'*dSdtheta2rr - sumprod_mpsimS*dSdtheta2rr + (multip*ooo).*dSdtheta2rr;
        
        % Subset version
        subxsimSx = x(index,:) - (ooo1*(subSimS'*x(index,:)));
        subxsimSxv = subxsimSx.*(ooo1*v(:,rr)'); 
        subdSdtheta2rr = (subSimS*ooo).*subxsimSxv;
           
        subdL2ddeltadthetarr = -subSimS*multip'*subdSdtheta2rr - subsumprod_mpsimS*subdSdtheta2rr + (multip*ooo).*subdSdtheta2rr;
        
        % Sum them
        dL2ddeltadtheta_index = dL2ddeltadtheta_index + pAll(tt)*dL2ddeltadthetarr/nn + pSub(tt)*subdL2ddeltadthetarr/nn;
        
        %% d2L of theta2 w.r.t. theta2
        
        % Full version
        dL2dtheta22rr = ((multip*ooo).*dSdtheta2rr)'*xsimSxv + sumprod_mpsimS*(-dSdtheta2rr'*x(index,:).*(ones(K+1,1)*v(:,rr)')) ;    
        
        % Subset version
        subdL2dtheta22rr = ((multip*ooo).*subdSdtheta2rr)'*subxsimSxv + subsumprod_mpsimS*(-subdSdtheta2rr'*x(index,:).*(ones(K+1,1)*v(:,rr)')) ; 
        
        % Sum them
        dL2dtheta22 = dL2dtheta22 + pAll(tt)*dL2dtheta22rr/nn + pSub(tt) * subdL2dtheta22rr/nn ;  
        
        %% d2L of delta & pi
        
        % Full version
        xsimSx1 = x(index,xindex) - (ooo1*(simS'*x(index,xindex)));
        xsimSxv1 = xsimSx1.*(ooo1*d(tt,rr)); 
        dSdpirr = (simS).*xsimSxv1;
           
        dL2ddeltadpirr = -simS*multip'*dSdpirr - sumprod_mpsimS*dSdpirr + (multip).*dSdpirr;
        
        % Subset version
        subxsimSx1 = x(index,xindex) - (ooo1*(subSimS'*x(index,xindex)));
        subxsimSxv1 = subxsimSx1.*(ooo1*d(tt,rr)); 
        subdSdpirr = (subSimS).*subxsimSxv1;
           
        subdL2ddeltadpirr = -subSimS*multip'*subdSdpirr - subsumprod_mpsimS*subdSdpirr + (multip).*subdSdpirr;
        
        % Sum them
        dL2ddeltadpi_index = dL2ddeltadpi_index + pAll(tt)*dL2ddeltadpirr/nn + pSub(tt)*subdL2ddeltadpirr/nn;
        
       %% d2L of pi & pi
        
        % Full version
        dL2dpirr = ((multip).*dSdpirr)'*xsimSxv1 + sumprod_mpsimS*(-dSdpirr'*x(index,xindex).*(d(tt,rr))) ;    
        
        % Subset version
        subdL2dpirr = ((multip).*subdSdpirr)'*subxsimSxv1 + subsumprod_mpsimS*(-subdSdpirr'*x(index,xindex).*(d(tt,rr))) ; 
        
        % Sum them
        dL2dpi2 = dL2dpi2 + pAll(tt)*dL2dpirr/nn + pSub(tt) * subdL2dpirr/nn ;  
        
       %% d2L of theta2 & pi
        
        % Full version
        dL2dpidtheta2rr = ((multip*ooo).*dSdtheta2rr)'*xsimSxv1 + sumprod_mpsimS*(-dSdtheta2rr'*x(index,xindex).*(ones(K+1,1)*d(tt,rr))) ;    
        
        % Subset version
        subdL2dpidtheta2rr = ((multip*ooo).*subdSdtheta2rr)'*subxsimSxv1 + subsumprod_mpsimS*(-subdSdtheta2rr'*x(index,xindex).*(ones(K+1,1)*d(tt,rr))) ; 
        
        % Sum them
        dL2dpidtheta2 = dL2dpidtheta2 + pAll(tt)*dL2dpidtheta2rr /nn + pSub(tt) * subdL2dpidtheta2rr /nn ;  
    end
        hess(2*K+3+index, 2*K+3+index) = dL2ddelta2DIAG;
        hess(2*K+3+index,K+2:2*K+2) = dL2ddeltadtheta_index;
        hess(K+2:2*K+2,2*K+3+index) = dL2ddeltadtheta_index';
        
        hess(2*K+3+index,2*K+3) = dL2ddeltadpi_index;
        hess(2*K+3, 2*K+3+index) = dL2ddeltadpi_index';
        
    end
    
    hess(K+2:2*K+2,K+2:2*K+2) = dL2dtheta22;    
    hess(2*K+3,2*K+3) =  dL2dpi2;
    hess(K+2:2*K+2, 2*K+3) = dL2dpidtheta2;
    hess(2*K+3, K+2:2*K+2) = dL2dpidtheta2';
