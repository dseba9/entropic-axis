% =========================================================================
% Dynamic Network Metrics: Sample Entropy of dSW and dFC
% =========================================================================
% Computes Dynamic Small-World Propensity (dSW) and Dynamic Functional 
% Connectivity (dFC) on fMRI sliding windows, and then applies Sample Entropy.

clc;
clear;

%% 1. Configuration & Paths
main_dir = '/home/usuario/disco2/proyectos/2024-LSD-fMRI';
addpath(fullfile(main_dir, 'scripts/tools/SWP(1)'));
addpath(fullfile(main_dir, 'scripts/tools/physionet.org/'));
addpath(fullfile(main_dir, 'scripts/tools/physionet.org/files/sampen/1.0.0/matlab/1.1-1'));

mask_name = 'Tian_Schaefer_combinada';
base_path = fullfile(main_dir, 'results', 'func_networks', mask_name, 'DATASET_NAME'); % UPDATE
output_dir = fullfile(main_dir, 'results', 'sample_entropy', 'Schaefer1000');
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

sub_list = [1, 2, 3]; % UPDATE with your subjects
task_conditions = {'COND1', 'COND2'}; % UPDATE
num_runs = 3;

% Graph & Entropy Parameters
thresholds = [95, 90, 85, 80, 75];
window_size = 24;
m = 2; 
r = 0.2;

%% 2. Initialization
SampEn_list_dSW = NaN(length(sub_list), length(task_conditions), num_runs, length(thresholds));
SampEn_list_dFC = NaN(length(sub_list), length(task_conditions), num_runs);

%% 3. Main Loop
for iSub = 1:length(sub_list)
    sub_id = sprintf('sub-%03d', sub_list(iSub));
    
    for iCond = 1:length(task_conditions)
        cond = task_conditions{iCond};
        
        for iRun = 1:num_runs
            file_name = sprintf('%s_ses-%s_task-rest_run-%02d_bold_%s.mat', sub_id, cond, iRun, mask_name);
            file_path = fullfile(base_path, sub_id, cond, file_name);
            if ~exist(file_path, 'file'), continue; end
            
            fprintf('Processing: %s %s R%d\n', sub_id, cond, iRun);
            data_s = load(file_path);
            func_roi = data_s.func_roi;
            clear data_s;
            
            % Handle Nans and Zeros
            valid_idx = find(~any(func_roi == 0, 2) & ~any(isnan(func_roi), 2));
            num_v = length(valid_idx);
            n_tp = size(func_roi, 2);
            num_w = n_tp - window_size + 1;
            
            % Hamming Window
            h_win = 0.54 - 0.46 * cos(2 * pi * (0:window_size-1) / (window_size-1));
            correlation_matrices = zeros(num_w, num_v, num_v);
            
            for w = 1:num_w
                chunk = func_roi(valid_idx, w:w+window_size-1);
                for ri = 1:num_v
                    v = chunk(ri, :);
                    if std(v) > 0, chunk(ri, :) = ((v - mean(v)) / std(v)) .* h_win; end
                end
                cm = corrcoef(chunk');
                cm(isnan(cm)) = 0; cm(logical(eye(size(cm)))) = 0; cm(cm < 0) = 0;
                correlation_matrices(w, :, :) = cm;
            end
            
            % Sliding Window Calculations
            swp_trace = zeros(num_w, length(thresholds));
            dfc_trace = zeros(num_w, 1);
            
            for w = 1:num_w
                cm = squeeze(correlation_matrices(w, :, :));
                dfc_trace(w) = mean(cm(cm > 0)); % dFC is the mean of positive connections
                
                for t = 1:length(thresholds)
                    th = prctile(cm(:), thresholds(t));
                    cm_th = cm; cm_th(cm < th) = 0;
                    swp_trace(w, t) = small_world_propensity(cm_th);
                end
            end
            
            % Sample Entropy
            s_fc = sampen(dfc_trace, m+1, r, 1);
            if ~isempty(s_fc), SampEn_list_dFC(iSub, iCond, iRun) = s_fc(3); end
            
            for t = 1:length(thresholds)
                s = sampen(swp_trace(:, t), m+1, r, 1);
                if ~isempty(s), SampEn_list_dSW(iSub, iCond, iRun, t) = s(3); end
            end
            
            clear correlation_matrices func_roi;
        end
    end
end

%% 4. Save Outputs
save(fullfile(output_dir, 'SampEn_list_dSW_GLOBAL.mat'), 'SampEn_list_dSW');
save(fullfile(output_dir, 'SampEn_list_dFC_GLOBAL.mat'), 'SampEn_list_dFC');
fprintf('Calculation complete.\n');
