%%%%%%%%%%%%%%%%%%%%%%%
% MAIN SCRIPT
%
% 1) Simulate Data from Random Coefficients Logit Demand model
% 2) Estimate model using MPEC
%
% All the files are adapted from Dube, Su, Fox(2012)
%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%% 
% SETTINGS
%%%%%%%%%%%
global share W PX x IV rc expmeanval% logshare
global datasort denomexpand sharesum oo
global T prods nn K v rc rctrue oo1 tol_inner tol_outer ConsPattern HessianPattern

%% ADD global variables 
global pAll pSub income phi std_income d  sub_index xindex
diary output_MPEC_Hessian_ktrlink.out;

randn('seed',355)                                   % reset normal random number generator
rand('seed',355)                                    % reset uniform random number generator
nn = 150;                                           % # draws to simulate shares
tol_inner = 1.e-14;
tol_outer = 1.e-6;
prods = 10;                                         % # products
T = 200;                                             % # Markets

%% Add probability of observing all choices or subset of choices and income

pAll = rand(T,1) * 0.2 + 0.6 ;  % Probability of observing all products
pSub = 1 - pAll; % Probability of observing first 5 products
income = 1.3 + randn(T,1) * 0.01 ; % Mean Income at each market
std_income = abs(0.1 + randn(T,1) * 0.01);
xindex = 1; % Location of pi in the Pi vector

starts = 1;                                         % # random start values to check during estimation

%% Export the data to csv

MktData2 = horzcat(transpose(1:T), pAll, pSub, income, std_income);  % Probability of choice sets data
csvwrite('MktData2.csv', MktData2)

% If you are given the data file, you would need to read the data file


%%%%%%%%%%%%%%%%%%%%%%%%
% TRUE PARAMETER VALUES
%%%%%%%%%%%%%%%%%%%%%%%%

phi = 1.3;  % Coefficient on income
costparams = ones(6,1)/2;
betatrue = [2 1.5 1.5 .5 -3]';                     % true mean tastes
betatrue0 = betatrue;
K = size(betatrue,1)-1;                             % # attributes
covrc = diag( [ .8 .9 .5 .5 .2] );                  % true co-variances in tastes
rctrue = covrc(find(covrc));
thetatrue = [betatrue;rctrue];
v = randn(length(betatrue),nn);                     % draws for share integrals during estimation
rc = chol(covrc)'*v;                                % draws for share integrals for data creation
sigmaxi=1;
covX = -.8*ones(K-1)+1.8*eye(K-1);                  % allow for correlation in X's
covX(1,3) = .3;
covX(2,3) = .3;
covX(3,1) = .3;
covX(3,2) = .3;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Create indices to speed share calculations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
oo = ones(1,nn);                                    % index for use in simulation of shares
oo1 = (0:T-1)*prods+prods;                          % indicates last observation for each market
sharesum = kron(speye(T), ones(1,prods));           % used to create denominators in predicted shares (i.e. sums numerators)
datasort = [(1:T*prods)' repmat((1:T)',prods,1) kron( (1:prods)',ones(T,1))];
datasort = datasort(:,1);
denomexpand = kron((1:T)',ones(prods,1));
sub_index = kron(0:prods:(T-1)* prods, ones(1,0.5* prods)) + kron(ones(1,T), 1:5); % index for first 5 products

%%%%%%%%%%%%%%%%%%%%%
% SIMULATE DATA
%%%%%%%%%%%%%%%%%%%%%
randn('seed',500)                                   % reset normal random number generator
rand('seed',500)                                    % reset uniform random number generator
xi = randn( T*prods,1)*sigmaxi;                       % draw demand shocks
A = [kron(ones(T,1), randn( prods,K-1)*chol(covX))];  % product attributes
prand = rand( T*prods,1)*5;
price = 3 +   xi*1.5 +  prand + sum(A,2);  % prices
z = rand( T*prods,length(costparams)) + 1/4*repmat( abs( prand +  sum(A,2)*1.1 ) ,1,length(costparams)); %instruments
x = [ones(T*prods,1) A price]; % x vector

income_draw = randn(1,nn);  % income shock draws
d = kron(ones(1,nn), income) + kron(ones(1,nn), std_income) .* kron(ones(T,1),income_draw); %simulate income
incomeExpand =  kron(income, ones(prods,1));    % expand income with number of products 

[share,nopurch] = mksharesim(betatrue,x,xi,rc); % market share simulation
csvwrite('MktData1.csv',[kron((1:T)', ones(prods,1)) kron(ones(T,1),(1:prods)') share x z]) % write the data to csv file
% Read you would need to read the data if given .csv file

z = horzcat(z, incomeExpand); % include mean income in z
y = log(share) - log(kron(nopurch,ones(prods,1)));     % log-odds ratio
iv = [ones(prods*T,1) A z ];                        % matrix of instruments

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2SLS ESTIMATION OF HOMOGENEOUS LOGIT MODEL %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xhat = iv*inv(iv'*iv)*iv'*x;
PX2sls = inv(xhat'*xhat)*xhat';                     % project. matrix on weighted x-space for 2SLS
beta2sls = PX2sls*y;
se2sls = sqrt(diag( mean((y-x*beta2sls).^2)*inv(xhat'*xhat) ));

expmeanval0 = exp(y);
IV = [ones(T*prods,1) A z A.^2  z.^2  prod(A,2) prod(z,2) kron(A(:,1),ones(1,size(z,2))).*z  kron(A(:,2),ones(1,size(z,2))).*z];
W = inv(IV'*IV);
PX = inv(x'*IV*W*IV'*x)*x'*IV*W*IV';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MPEC ESTIMATION OF RANDOM COEFFICIENTS LOGIT MODEL %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

theta20 = 0.5*abs(beta2sls);
startvalues = repmat(theta20',starts,1).* cat(2,ones(size(theta20)),rand([starts-1,size(theta20',2)] )' *1 )';

nIV = size(IV,2);       % # instrumental variables
Pi = 1.1; % initial guess for coefficient on income

GMPEC = 1.0e20;
CPUtMPEC = 0;
FuncEvalMPEC = 0;
GradEvalMPEC = 0;
HessEvalMPEC = 0;
expmeanval = expmeanval0;


for reps=1:starts,
    
    theta20 = startvalues(reps,:)';
    delta0 = invertshares(theta20, Pi);
    theta10 = PX*delta0;
    Pi0 = 1.1;
    resid0 = delta0 - x*theta10;
    g0 = IV'*resid0;
    x0 = [theta10; theta20; Pi0; delta0; g0];
    
    x_L = -Inf*ones(length(x0),1);   % Lower bounds for x.
    x_L(6:10) = 0;
    x_L(11) = 1;
    x_U =  Inf*ones(length(x0),1);   % Upper bounds for x.
    
    neF    = prods*T + size(IV, 2);    
    nx0 = size(x0,1);
    
    disp('Start of Constructing sparsity pattern of constraint Jacobian')
    
    % Sparsity patterns of constraints, 0 if derivative 0, 1 otherwise
    % Derivatives of market shares
    
    c11 = zeros(T*prods, K+1);   % market shares with respect to mean parameters
    c12 = ones(T*prods, K+1);  % market shares with respect to diagonals of Sigma
    c13 = ones(T*prods, 1); % market shares with respect to interaction with income    
    c14 = kron(eye(T), ones(prods)); % market shares with respect to mean utilities
    c15 = zeros(T*prods, nIV); % market shares with respect to moment values
    
    % Derivatives of moments
    
    c21 = ones(nIV, K + 1);   % moments with respect to mean parameters 
    c22 = zeros(nIV, K + 1);    % moments with respect to diagonals of Sigma
    c23 = zeros(nIV, 1);  % moments with respect to interaction with demo    
    c24 = ones(nIV, T*prods);  % moments with respect to mean utilities
    c25 = eye(nIV);       % moments with respect to moment values 

    
    ConsPattern = [ c11 c12 c13 c14 c15; c21 c22 c23 c24 c25];   
       
    disp('End of Constructing sparsity pattern of constraint Jacobian')
    
    disp('Start of Constructing sparsity pattern of Hessian of the Lagragian')
    
    HessianPattern = zeros(nx0, nx0);  % Hessian for d2L 
    HessianPattern(2*K+4+T*prods:end, 2*K+4+T*prods:end) = ones(nIV, nIV); % Hessian for d2L w.r.t. g & g
    HessianPattern(K+2:2*K+3,K+2:2*K+3+T*prods) = ones(K+2, K+2+T*prods); % Hessianl for d2L w.r.t.  (theta2 & Pi) (theta2 & Pi)  
    HessianPattern(2*K+4:2*K+3+T*prods,K+2:2*K+3) = ones(T*prods, K+2); % Hessian for d2L w.r.t. delta (theta2 & Pi)
    for tt=1:T,
        index = 2*K+3+(1:prods)'+(tt-1)*prods;  % Hessian for d2L delta wr.t. delta
        HessianPattern(index, index) = ones(prods, prods);
    end

    disp('END of Constructing sparsity pattern of Hessian of the Lagragian')  
    
    ktropts = optimset('DerivativeCheck','on','Display','iter',...
           'GradConstr','on','GradObj','on','TolCon',1E-6,'TolFun',1E-6,'TolX',1E-6,'JacobPattern',ConsPattern,'Hessian','user-supplied','HessFcn',@GMMMPEC_hess_sparse_ktr, 'HessPattern', HessianPattern);  
   
    t1 = cputime;   
       
    [X fval_rep exitflag output lambda] = ktrlink(@GMMMPEC_f_ktr,x0,[],[],[],[],x_L,x_U,@GMMMPEC_c_sparse_ktr,ktropts, 'knitroOptions2.opt');    
    
    CPUtMPEC_rep = cputime - t1;
    
    theta1MPEC_rep = X(1:K+1);
    theta2MPEC_rep = X(K+2:2*K+2);
    GMPEC_rep = fval_rep;
    INFOMPEC_rep = exitflag;
    
    CPUtMPEC = CPUtMPEC + CPUtMPEC_rep;    
    FuncEvalMPEC = FuncEvalMPEC + output.funcCount;
    GradEvalMPEC = GradEvalMPEC + output.funcCount+1;
    HessEvalMPEC = HessEvalMPEC + output.funcCount;
    
    if (GMPEC_rep < GMPEC && INFOMPEC_rep==0),
         thetaMPEC1 = theta1MPEC_rep;
         thetaMPEC2 = theta2MPEC_rep;
         GMPEC = GMPEC_rep;
         INFOMPEC = INFOMPEC_rep;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COVARIANCE MATRIX FOR MPEC STRUCTURAL PARAMETERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

delta = X(2*K+4:2*K+3+prods*T);    % mean utilities
resid = delta - x*thetaMPEC1;  % xi
Ddelta = jacobian(X);       % jacobian matrix of mean utilities
covg = zeros(size(IV,2));
for ii =1:length(IV),
    covg = covg + IV(ii,:)'*IV(ii,:)*(resid(ii)^2);
end
Dg = [x Ddelta]'*IV;            % gradients of moment conditions
covMPEC = inv( Dg*W*Dg')*Dg*W*covg*W*Dg'*inv( Dg*W*Dg');

results = [ [thetaMPEC1; thetaMPEC2 ;X(2*K + 3)] covMPEC]; 
csvwrite('results,csv',results)
diary off;