function bands = get_band_subset(groups, data2D)
% The nearest band to cluster center is selected as
% representative band according to clustering result.

bands = [];
for i = 1:length(unique(groups))
    idx = find(groups == i);
    band_i = data2D(:,idx)';
    center = mean(band_i, 1);
    dist = EuDist2(band_i, center);
    [~, rep] = min(dist);
    bands = [bands idx(rep)];
end

