clc;clear all;close all;
addpath(genpath(cd));
rng(7);

%% load dataset
load Indian_pines_corrected.mat;load Indian_pines_gt.mat;
data_src = indian_pines_corrected;data_gt = indian_pines_gt;
clear indian_pines_corrected;clear indian_pines_gt;
Ratio = 0.0812;% the value of Ratio can be obtained by the function of "Edge_ratio3" in the function of "cubseg"

[H, W, B] = size(data_src);
data_src = double(data_src);

% smoothing
for i=1:B
    data3D(:,:,i) = imfilter(data_src(:,:,i),fspecial('average'));
end

%% obtain global & local graph
% superpixel segmentation
SegPara = 2000;
labels = cubseg(data3D, SegPara * Ratio);
viewNum = length(unique(labels));

for i = 1:B
    img(:,:,i) = (data3D(:,:,i)-min(min(data3D(:,:,i)))) /...
        (max(max(data3D(:,:,i)))-min(min(data3D(:,:,i))));
end
data2D = reshape(img, H * W, B);

X = cell(1, viewNum);
A = cell(1, viewNum);
X{viewNum} = [];
k = 5;
%compute global graph
for v=1:viewNum - 1
    idx = find(labels == v);
    X{v} = data2D(idx,:);

    X{viewNum} = [X{viewNum};mean(X{v},1)];

    distX = L2_distance_1(X{v}',X{v}');
    %distX = sqrt(distX);
    [distX1, idx2] = sort(distX,2); %sort each row, increase
    num = size(X{v},1);
    A{v} = zeros(num);
    rr = zeros(num,1);
    for i = 1:num
        di = distX1(i,2:k+2); %Exclude itself
        rr(i) = 0.5*(k*di(k+1)-sum(di(1:k)));
        id = idx2(i,2:k+2);
        A{v}(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
    end
end

%compute local graph
distX = L2_distance_1(X{viewNum}',X{viewNum}');
%distX = sqrt(distX);
[distX1, idx2] = sort(distX,2); %sort each row, increase
num = size(X{viewNum},1);
A{viewNum} = zeros(num);
rr = zeros(num,1);
for i = 1:num
    di = distX1(i,2:k+2); %Exclude itself
    rr(i) = 0.5*(k*di(k+1)-sum(di(1:k)));
    id = idx2(i,2:k+2);
    A{viewNum}(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
end

for v = 1:viewNum
    XTATAX{v} = X{v}'*A{v}'*A{v}*X{v};
    XTATX{v} = X{v}'*A{v}'*X{v};
end
%% optimization

lambda = 1e4; p = 1;
sz = [B B viewNum];
weight_vector = ones(1,viewNum)';

Z_tensor = zeros(sz);
Y_tensor = zeros(sz);
W_tensor = zeros(sz);

alpha = zeros(viewNum, 1);
for v = 1:viewNum
    alpha(v) = 1 / viewNum;
end
rho = 1;eta = 2;rho_max = 1e6;
iter = 0;
max_iter = 20;
while iter < max_iter
    for v = 1:viewNum
        %update W_v and alpha_v
        We{v} = inv(2/alpha(v)*XTATAX{v}+rho*eye(B))*(rho*Z_tensor(:,:,v)...
            -Y_tensor(:,:,v)+2/alpha(v)*XTATX{v});
        alpha(v) = norm(X{v}-A{v}*X{v}*We{v},'fro');
    end
    alpha = alpha / sum(alpha);
    %update Z_tensor
    W_tensor = cat(3, We{:,:});
    for v = 1:viewNum
        QQ{v} = (We{v}+Y_tensor(:,:,v)/rho);
    end
    Q_tensor = cat(3, QQ{:,:});
    q = Q_tensor(:);
    [z, objV] = wshrinkObj_weight_lp(q, weight_vector*lambda/rho, sz, 0, 3, p);
    Z_tensor = reshape(z, sz);
    %update multiplier and penalty parameter
    Y_tensor = Y_tensor + rho*(W_tensor - Z_tensor);
    rho = min(rho*eta,rho_max);
    iter = iter + 1;
end

%% calculation of affinity matrix
A_res = zeros(B,B);
for v= 1:viewNum
    A_res = A_res + (abs(W_tensor(:,:,v)) + abs(W_tensor(:,:,v)'))/viewNum;
end

%% band selection
band = 20;
C = SpectralClustering(A_res,band);
Y = get_band_subset(C, data2D);
Y = sort(Y)

