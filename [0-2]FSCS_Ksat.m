%Please note that after comparison, we found that the accelerated Python
%implementation based on the scikit-learn-ex library achieved the best
%results with large sample sizes. Please refer to the manuscript A2 for
%details. However, we have retained the current MATLAB code to illustrate
%the implementation of FSCS sampling in the original study.


clc;
clear;

% Read data
grd = readtable('sampleddata/USKSAT_OpenRefined_cleaned.csv');

% Remove the first column and Ksat
index = grd{:, 1};
ksat = grd{:, 2};
grd_data = grd{:, 3:end};

% Standardize features
grd_scaled = zscore(grd_data);

% Define sample sizes
sample_levels = [1000, 5000, 10000];
num_samples_per_size = 20;

% Loop to generate sample sets of different sizes and repetitions
for level_idx = 1:length(sample_levels)
    level = sample_levels(level_idx);
    
    % Create parallel pool
    if level == 10000 || level == 20000
        parpool('local', 5);
    else
        parpool('local', 5);
    end
    
    for t = 1:num_samples_per_size
        sample_file = sprintf('sampleddata/combined_samples_Ksat/FSCS_sampled_data_%d_set_%d.csv', level, t);
        
        if isfile(sample_file)
            fprintf('File %s already exists, skipping...\n', sample_file);
            continue;
        end
        
        % Set parallel computation options
        options = statset('UseParallel', 1);

        % Perform k-means++ clustering
        [idx, C, sumd] = kmeans(grd_scaled, level, 'MaxIter', 1000, 'Replicates', 5, 'Start', 'plus', 'Options', options);

        % Get cluster centers
        cluster_centers = C;

        % Find the closest data point to each cluster center
        kmeans_sampled_indices = zeros(level, 1);
        for i = 1:level
            distances = sum((grd_scaled - cluster_centers(i, :)).^2, 2);
            [~, kmeans_sampled_indices(i)] = min(distances);
        end

        % Get the sampled data
        kmeans_sampled_df = grd(kmeans_sampled_indices, :);

        % Save the data
        writetable(kmeans_sampled_df, sample_file);
        fprintf('Generated sample set %d for sample size %d\n', t, level);
    end
    
    % Close parallel pool
    delete(gcp('nocreate'));
end