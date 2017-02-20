function [share,nopurch] = mksharesim(betatrue,x,xi,rc)

%%%%%%%%%%%%
% MKSHARESIM
% generates market share data.


global prods T datasort pAll pSub phi  d nn xindex

delta = x*betatrue+xi;

if nargin==4,
    MU = x*rc + kron(x(:,xindex), ones(1,nn)) .* kron(d, ones(prods,1)) * phi ;
    numer = exp( repmat(delta,1,size(rc,2)) + MU );
else
    numer = exp(delta);
end
temp = cumsum(numer);
temp1 = temp( (1:T)*prods,: );
temp1(2:end,:) = diff(temp1);
denom = 1+temp1; % denominator if you observe all products
% share = mean(numer./repmat(denom,prods,1),2);

%% CHANGE share simulation process

add_temp1 = temp( (1/2:1/2:T)*prods, :);    % extract every 1/2*prods in each market of cumsum
add_temp1(2:end, :) = diff(add_temp1);  % this is the sum of exp(utility) for first 5 firm in the market
add_denom = add_temp1(1:2:end,:) + 1;   % denominator if observe only first 5 firms in each market
add_numer = zeros(size(numer,1), size(numer,2));
add_index = kron(0:prods:(T-1)* prods, ones(1,0.5* prods)) + kron(ones(1,T), 1:5); % matrix index for first 5 firm
add_numer(add_index, :) = numer(add_index,:); % Set firms/prod 5 to 10 to zero
share = mean(numer./ kron(denom ./ kron(ones(1,nn),pAll) ,ones(prods,1)),2) ...
    + mean(add_numer./kron(add_denom ./ kron(ones(1,nn),pSub) , ones(prods,1)),2);


nopurch = mean(kron(ones(1,nn),pAll) ./denom + kron(ones(1,nn),pSub) ./add_denom,2) ;
 