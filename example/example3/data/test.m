clear;
clc;

a = importdata('response_frame12_ida_comb.csv');
gmtag = [find(a.data(:,1)==0.1);length(a.data(:,1))+1];

sa = [0.1690,0.2594,0.3696,0.5492,0.7131,0.9000];
for i = 1:1:length(gmtag)-1
    cur_data = a.data(gmtag(i):gmtag(i+1)-1,:);
    % interporlation
    for j = 1:1:length(sa)
        try
            tmp = interp1(cur_data(:,1),cur_data(:,2:end),sa(j));
            if isnan(tmp(2))
                disp('Exceeding the range, no data found');
            else
                sida{j,1}(i,:) = [sa(j),tmp];
            end
        catch
            disp('Exceeding the range, no data found');
        end
    end
end

metadata = [];
for i = 1:1:length(sa)
    metadata = [metadata;sida{i,1}(sida{i,1}(:,1)>0,:)];
end