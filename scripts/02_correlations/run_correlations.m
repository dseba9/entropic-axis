function run_correlations()
    % RUN_CORRELATIONS Standardized pipeline to calculate functional connectivity
    % matrices from parcellated BOLD fMRI time-series.
    %
    % It loads the extracted regional BOLD time-series (.mat files),
    % computes the Pearson correlation matrix for each subject/session,
    % zeroes out the diagonal, and exports the correlation tensor.
    
    clc;
    clear;
    
    %% ── CONFIGURATION ────────────────────────────────────────────────────────
    script_dir = fileparts(mfilename('fullpath'));
    main_dir = fileparts(fileparts(script_dir)); % Walk up to GITHUB_ENTROPIC_AXIS
    
    % Path to SPM12 (adjust if needed)
    spm_path = '/home/usuario/disco2/toolboxes/matlab/spm12';
    
    % Select Atlas to load
    mask_name = 'AAL'; % Options: 'AAL', 'Tian_Schaefer_combinada'
    
    % Define dataset details
    sub_list = [1, 2, 3, 4, 6, 9, 10, 11, 12, 13, 15, 17, 18, 19, 20];
    task_conditions = {'LSD', 'PLCB'};
    num_runs = 3;
    num_rois = 90; % Update to match atlas (e.g. 90 for AAL, 1000 for Schaefer)
    
    %% ── INITIALIZATION ───────────────────────────────────────────────────────
    if exist(spm_path, 'dir')
        addpath(spm_path);
    end
    
    input_dir = fullfile(main_dir, 'results', 'func_networks', mask_name);
    output_dir = fullfile(input_dir, 'correlation_matrices');
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    % Initialize tensor to store correlation matrices
    correlation_matrices = zeros(length(sub_list), length(task_conditions), num_runs, num_rois, num_rois);
    
    %% ── COMPUTATION ──────────────────────────────────────────────────────────
    fprintf('=== Starting Correlation Matrix Calculation ===\n');
    
    for iSub = 1:length(sub_list)
        sub_id = sprintf('sub-%03d', sub_list(iSub));
        for iCond = 1:length(task_conditions)
            cond_name = task_conditions{iCond};
            for iRun = 1:num_runs
                
                % Build file path to parcellated BOLD data
                file_path = fullfile(input_dir, sub_id, cond_name, ...
                    sprintf('%s_ses-%s_task-rest_run-%02d_bold_%s.mat', sub_id, cond_name, iRun, mask_name));
                
                if ~exist(file_path, 'file')
                    fprintf('[SKIP] Parcellated file not found: %s\n', file_path);
                    continue;
                end
                
                % Load parcellated data (loads 'func_roi' variable)
                data_load = load(file_path);
                func_roi = data_load.func_roi;
                
                % Replace NaNs with zero if any
                func_roi(isnan(func_roi)) = 0; 
                
                % Calculate Pearson correlation matrix
                correlation_matrix = corrcoef(func_roi');
                
                % Zero out the diagonal (self-connections)
                correlation_matrix(logical(eye(size(correlation_matrix)))) = 0; 
                
                % Store in the tensor
                correlation_matrices(iSub, iCond, iRun, :, :) = correlation_matrix; 
            end
        end
    end
    
    % Save correlation tensor to output directory
    save_path = fullfile(output_dir, 'correlation_matrices.mat');
    save(save_path, 'correlation_matrices');
    fprintf('Correlation matrices successfully saved to %s\n', save_path);
    fprintf('=== Calculation Finished ===\n');
end
