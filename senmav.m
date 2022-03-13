function [U,time]=senmav(Y, q, lambda)

%% [U,time]=senmav(Y, q, lambda)
%
%  Spatial ENergy Prior Constrained MAximum Simplex Volume (SENMAV)
%
% ------------------------------------------------------------------------
%%  ===== Required inputs =============
%   
%   Y - matrix with  nr(rows) x nc(columns) x nb (bands).
%   q - number of endmembers
%   lambda - regularization parameter
%
%%  =========================== Outputs ==================================
%
% U  =  endmember matrix
%
% time =  computational time
%
%
%% -------------------------------------------------------------------------
%
% Copyright (May, 2020):    Wenxing Bao (bwx71@163.com)
%                           Xiangfei Shen (xfshen95@163.com)
%
%
% Permission to use, copy, modify, and distribute this software for
% any purpose without fee is hereby granted, provided that this entire
% notice is included in all copies of any software which is or includes
% a copy or modification of this software and in all copies of the
% supporting documentation for such software.
% This software is being provided "as is", without any express or
% implied warranty.  In particular, the authors do not make any
% representation or warranty of any kind concerning the merchantability
% of this software or its fitness for any particular purpose."
% ----------------------------------------------------------------------
%
% More details in:
%
% [1] Xiangfei Shen, Wenxing Bao, Kewen Qu. Spatial-spectral hyperspectral 
% endmember extraction using a spatial energy prior constrained maximum simplex 
% volume approach[J]. IEEE Journal of Selected Topics in Applied Earth Observations 
% and Remote Sensing, 2020, 13: 1347-1361.
%
% [2] Xiangfei Shen, Wenxing Bao, Kewen Qu. Subspace-based preprocessing 
% module for fast hyperspectral endmember selection[J]. IEEE Journal of Selected 
% Topics in Applied Earth Observations and Remote Sensing, 2021, 14:3386-3402.
%
% -------------------------------------------------------------------------

%%
tic

[nr,nc,~]=size(Y);

HIM=hyperConvert2d(Y);

[m, n]=ind2sub([nr, nc], 1:nr*nc);

odi=[m',n'];

k=q+floor(q/2);

%%

[hsi, ~, ~]=hyperPct(HIM,q-1);

%%

[class, centroid] = kmeans(hsi',k,'MaxIter', 200);

class=reshape(class,nr,nc);

C = padarray(class,[1 1],'symmetric','both');


%% initialization


inidx=randperm(k,q);

U_idx =  inidx;% Random endmember selection
E     = centroid(U_idx,:)';    % Endmember matrix

J0 = abs(det([ones(1,q); E])) / factorial(q-1); % Simplex volume

J  = zeros(q,1);

for j = 1:nr*nc;
    
    for k = 1:q;
        E_tmp      = E;
        E_tmp(:,k) = hsi(:,j);
        J(k)=abs(det([ones(1,q); E_tmp])) / factorial(q-1);
    end
    
    [V_tmp,k_idx] = max(J);
    
    if V_tmp > J0
        J0            = V_tmp;
        E(:,k_idx)   = hsi(:,j);
        U_idx(k_idx) = j;
    end
    
end



%% SSCEE


alpha=10^(-floor(log10(J0))-1);
U_idx=inidx;
E     = centroid(U_idx,:)';    % Endmember matrix

J0 = abs(det([ones(1,q); E])) / factorial(q-1); % Simplex volume

J  = zeros(q,1);

for j = 1:nr*nc;
    
    for k = 1:q;
        E_tmp      = E;
        index_tmp  = U_idx;
        E_tmp(:,k) = hsi(:,j);
        index_tmp(k)=j;
        J(k)    = alpha*(abs(det([ones(1,q); E_tmp])) ...
            / factorial(q-1) ) +lambda*energyprior(index_tmp, C, odi) ;
    end
    
    [V_tmp,k_idx] = max(J);
    if V_tmp > J0
        J0            = V_tmp;
        E(:,k_idx)   = hsi(:,j);
        U_idx(k_idx) = j;
    end
    
end



U = HIM(:, U_idx);


time=toc;
end

function energy=energyprior(U_idx, C, odi)

e=zeros(length(U_idx),1);
xy=odi(U_idx,:);
for i=1:length(U_idx)
    sig=0;
    c=C(xy(i,1):xy(i,1)+2,xy(i,2):xy(i,2)+2);
    c=reshape(c,3*3,1);
    cent=c(5,:);
    c(5,:)=[];
    for j=1:length(c)
        if cent==c(j)
            sig=sig+0;
        else
            sig=sig+1;
        end
    end
    e(i)=exp(-sig);
    
end
energy=mean(e);
end