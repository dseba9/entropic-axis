% =========================================================================
% Leave-One-Network-Out (LONO) Analysis for dSW and dFC
% =========================================================================
% Iteratively removes one predefined resting-state network at a time from
% the functional connectivity matrix before recalculating graph metrics.

clc;
clear;

%% 1. Configuration & Paths
main_dir = '/home/usuario/disco2/proyectos/2024-LSD-fMRI';
addpath(fullfile(main_dir, 'scripts/tools/SWP(1)'));
addpath(fullfile(main_dir, 'scripts/tools/physionet.org/'));
addpath(fullfile(main_dir, 'scripts/tools/physionet.org/files/sampen/1.0.0/matlab/1.1-1'));

mask_name = 'Tian_Schaefer_combinada';
base_path = fullfile(main_dir, 'results', 'func_networks', mask_name, 'DATASET_NAME'); % UPDATE
output_dir = fullfile(main_dir, 'results', 'sample_entropy', 'Schaefer1000', 'LONO');
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

sub_list = [1, 2, 3]; % UPDATE
task_conditions = {'COND1', 'COND2'}; % UPDATE
num_runs = 3;

thresholds = [95, 90, 85, 80, 75];
window_size = 24; m = 2; r = 0.2;
original_n_rois = 1000;

%% 2. Load Yeo 7-Networks Look-Up Table
lut_path = fullfile(main_dir, 'scripts/01_parcelling/masks/1000_Schaefer/Schaefer2018_1000Parcels_7Networks_order.lut');
fileID = fopen(lut_path, 'r');
lut_data = textscan(fileID, '%d %f %f %f %s');
fclose(fileID);
names = lut_data{5};

ROI_to_net = zeros(original_n_rois, 1);
network_names = {'Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default'};
num_networks = length(network_names);

for i = 1:original_n_rois
    parts = strsplit(names{i}, '_');
    if length(parts) >= 3
        idx = find(strcmp(network_names, parts{3}));
        if ~isempty(idx), ROI_to_net(i) = idx; end
    end
end

%% 3. Initialization
SampEn_LONO_dSW = NaN(length(sub_list), length(task_conditions), num_runs, num_networks, length(thresholds));
SampEn_LONO_dFC = NaN(length(sub_list), length(task_conditions), num_runs, num_networks);

%% 4. Main LONO Loop
for iSub = 1:length(sub_list)
    sub_id = sprintf('sub-%03d', sub_list(iSub));
    
    for iCond = 1:length(task_conditions)
        cond = task_conditions{iCond};
        
        for iRun = 1:num_runs
            file_name = sprintf('%s_ses-%s_task-rest_run-%02d_bold_%s.mat', sub_id, cond, iRun, mask_name);
            file_path = fullfile(base_path, sub_id, cond, file_name);
            if ~exist(file_path, 'file'), continue; end
            
            fprintf('Processing LONO: %s %s R%d\n', sub_id, cond, iRun);
            data_s = load(file_path);
            func_roi = data_s.func_roi;
            clear data_s;
            
            % Subcortical clipping (LONO is only for cortical 1000 ROIs)
            if size(func_roi, 1) > 1000, func_roi = func_roi(1:1000, :); end
            
            valid_idx = find(~any(func_roi == 0, 2) & ~any(isnan(func_roi), 2));
            num_v = length(valid_idx);
            n_tp = size(func_roi, 2);
            num_w = n_tp - window_size + 1;
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
            
            % Map valid ROIs to their respective networks
            valid_nets = ROI_to_net(valid_idx);
            
            for net_idx = 1:num_networks
                to_rem = find(valid_nets == net_idx);
                swp_trace = zeros(num_w, length(thresholds));
                dfc_trace = zeros(num_w, 1);
                
                for w = 1:num_w
                    cm_loo = squeeze(correlation_matrices(w, :, :));
                    % LEAVE ONE NETWORK OUT: Remove rows and columns of this network
                    cm_loo(to_rem, :) = []; cm_loo(:, to_rem) = [];
                    
                    dfc_trace(w) = mean(cm_loo(cm_loo > 0));
                    for t = 1:length(thresholds)
                        th = prctile(cm_loo(:), thresholds(t));
                        cm_th = cm_loo; cm_th(cm_loo < th) = 0;
                        swp_trace(w, t) = small_world_propensity(cm_th);
                    end
                end
                
                % Sample Entropy Calculation
                s_fc = sampen(dfc_trace, m+1, r, 1);
                if ~isempty(s_fc), SampEn_LONO_dFC(iSub, iCond, iRun, net_idx) = s_fc(3); end
                
                for t = 1:length(thresholds)
                    s = sampen(swp_trace(:, t), m+1, r, 1);
                    if ~isempty(s), SampEn_LONO_dSW(iSub, iCond, iRun, net_idx, t) = s(3); end
                end
            end
            
            clear correlation_matrices func_roi;
        end
    end
end

%% 5. Save Outputs
save(fullfile(output_dir, 'SampEn_list_LONO_dSW.mat'), 'SampEn_LONO_dSW');
save(fullfile(output_dir, 'SampEn_list_LONO_dFC.mat'), 'SampEn_LONO_dFC');
fprintf('LONO Calculation complete.\n');
