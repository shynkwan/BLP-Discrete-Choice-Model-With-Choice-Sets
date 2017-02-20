function [subEstShare, subSimShare] = sub_ind_shnormMPEC(expmeanval,expmu)

% IND_SHNORM
% This function computes the distribution of "individual" probabilities of choosing each brand.
% The probabilities are those associated with the normally-distributed r.c. logit.
% Individuals only observe a subset of the products
%
% ARGUMENTS:
% expmeanval = vector of exponentiated mean utilities, sorted by market then product
% expmu = matrix of exponentiated draws of deviations from mean utilities, sorted by market then product
%
% OUTPUT:
% PROB = vector of expected market shares, sorted by market then product
%
% source: Dube, Fox and Su (2008)
% Code Revised: April 2008

global oo sharesum denomexpand sub_index

numer = (expmeanval*oo ).*expmu;        % this is the numerator (oo speeds-up expanding mean utility by number of draws)
add_numer = zeros(size(numer,1), size(numer,2));
add_numer(sub_index, :) = numer(sub_index,:); % Set firms/prod 5 to 10 to zero

sum1 = sharesum*add_numer;
sum11 = 1./(1+sum1);                    % this is the denominator of the shares
denom1 = sum11(denomexpand,:);          % this expands the denominator
subSimShare = add_numer.*denom1;                     % simulated shares for each draw
subEstShare = mean(subSimShare,2);                      % expected share (i.e. mean across draws)