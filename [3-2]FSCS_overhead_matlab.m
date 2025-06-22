clc;
clear;
rng(2025); % Ensure reproducibility

% Read data
grd = readtable('sampleddata/class_df.csv');
grd_data = grd{:, 2:end-1};
grd_scaled = zscore(grd_data); % Standardize features

% Parameter settings
raw_sizes = 10000:10000:100000;         % Raw sample sizes
sample_levels = 5000:5000:65000;        % Sample sizes for clustering
feature_nums = 1:10;                    % Number of features
replicates = 5;                         % Number of random feature combinations

% Resume from breakpoint: read completed results
if isfile('FSCS_overhead_with_rawsize_temp.csv')
    results = readtable('FSCS_overhead_with_rawsize_temp.csv');
    done = [results.raw_n, results.feat_num, results.level, results.comb_idx];
else
    results = [];
    done = [];
end

for raw_n = raw_sizes
    for feat_num = feature_nums
        if feat_num > size(grd_scaled,2)
            continue;
        end

        % Randomly select 'replicates' different feature combinations
        if feat_num < size(grd_scaled,2)
            all_combs = nchoosek(1:size(grd_scaled,2), feat_num);
            comb_count_all = size(all_combs, 1);
            if comb_count_all <= replicates
                rand_idx = 1:comb_count_all;
            else
                rand_idx = randperm(comb_count_all, replicates);
            end
            feat_combs = all_combs(rand_idx, :);
            comb_count = size(feat_combs, 1);
        else
            feat_combs = 1:size(grd_scaled,2);
            comb_count = 1;
        end

        for level = sample_levels
            if level > raw_n
                continue;
            end
            for comb_idx = 1:comb_count
                % Check if the current combination is already completed
                if ~isempty(done)
                    is_done = any(done(:,1)==raw_n & done(:,2)==feat_num & ...
                        done(:,3)==level & done(:,4)==comb_idx);
                    if is_done
                        continue;
                    end
                end
                try
                    idx_raw = randperm(size(grd_scaled,1), raw_n);
                    data_raw = grd_scaled(idx_raw, :);
                    feat_idx = feat_combs(comb_idx, :);
                    data_sub = data_raw(:, feat_idx);

                    % Memory sampling initialization
                    mem_samples = [];
                    keep_sampling = true;
                    t = timer('ExecutionMode','fixedSpacing','Period',0.1, ...
                        'TimerFcn',@(~,~) assignin('base','mem_samples',[evalin('base','mem_samples'), memory().MemUsedMATLAB]), ...
                        'StopFcn',@(~,~) assignin('base','keep_sampling',false));
                    assignin('base','mem_samples',mem_samples);
                    assignin('base','keep_sampling',keep_sampling);

                    start(t);
                    t_start = tic;
                    [idx, C, sumd] = kmeans(data_sub, level, ...
                        'MaxIter', 100, ...
                        'Replicates', 1, ...
                        'Start', 'plus');
                    elapsed_time = toc(t_start);
                    stop(t);
                    delete(t);

                    mem_samples = evalin('base','mem_samples');
                    if isempty(mem_samples)
                        peak_mem_used = NaN;
                    else
                        peak_mem_used = max(mem_samples)/1024/1024; % MB
                    end

                    results = [results;
                        table(raw_n, feat_num, level, comb_idx, elapsed_time, peak_mem_used)];
                    if mod(height(results), 100) == 0
                        writetable(results, 'FSCS_overhead_with_rawsize_temp.csv');
                    end
                    fprintf('RawN=%d, Feat=%d, Level=%d, Comb=%d, Time=%.2fs, PeakMem=%.2fMB\n', ...
                        raw_n, feat_num, level, comb_idx, elapsed_time, peak_mem_used);
                    clear idx C sumd data_sub data_raw feat_idx mem_samples t;
                catch ME
                    warning('Error at RawN=%d, Feat=%d, Level=%d, Comb=%d: %s', ...
                        raw_n, feat_num, level, comb_idx, ME.message);
                    % Skip errors such as out of memory
                    if contains(ME.message, 'memory', 'IgnoreCase', true) || ...
                            contains(ME.message, 'Out of memory', 'IgnoreCase', true)
                        continue;
                    end
                end
            end
        end
    end
end

writetable(results, 'FSCS_overhead_with_rawsize.csv');