function run_parcelling()
    % RUN_PARCELLING Standardized pipeline to extract BOLD time-series from fMRI data.
    % This script unifies all dataset-specific parcellation scripts.
    % It loads a NIfTI volume, applies a parcellation mask, averages BOLD
    % signals within each ROI, and saves the extracted time-series.
    %
    % Requirements: SPM12 must be in the MATLAB path.
    
    clc;
    clear;
    
    %% ── CONFIGURATION ────────────────────────────────────────────────────────
    % 1. Define paths
    script_dir = fileparts(mfilename('fullpath'));
    main_dir = fileparts(fileparts(script_dir)); % Walk up to GITHUB_ENTROPIC_AXIS
    
    % Path to SPM12 (adjust if needed)
    spm_path = '/home/usuario/disco2/toolboxes/matlab/spm12';
    
    % 2. Select Dataset to process:
    % Options: 'LSD', 'DMT', 'Anesthesia', 'Modafinil', 'Schizophrenia'
    dataset = 'LSD'; 
    
    % 3. Select Atlas Mask:
    % Options: 'AAL' (90 ROIs), 'Tian_Schaefer_combinada' (Schaefer 1000 + Tian Subcortex)
    atlas_name = 'AAL'; 
    
    %% ── INITIALIZATION ───────────────────────────────────────────────────────
    addpath(script_dir); % Add parcellation helper functions
    if exist(spm_path, 'dir')
        addpath(spm_path);
    else
        warning('SPM12 path not found. Ensure SPM12 is added to the MATLAB path.');
    end
    
    fprintf('=== Starting Parcellation Pipeline ===\n');
    fprintf('Dataset: %s\n', dataset);
    fprintf('Atlas:   %s\n', atlas_name);
    
    % Load Mask header and data
    mask_file = fullfile(script_dir, 'masks', atlas_name, [atlas_name, '.nii']);
    if ~exist(mask_file, 'file')
        error('Atlas mask NIfTI file not found at: %s', mask_file);
    end
    
    mask_hdr = spm_vol(mask_file);
    mask_data = spm_read_vols(mask_hdr);
    
    % Define subject list and conditions based on dataset
    switch dataset
        case 'LSD'
            sub_list = [1, 2, 3, 4, 6, 9, 10, 11, 12, 13, 15, 17, 18, 19, 20];
            conditions = {'LSD', 'PLCB'};
            runs = 1:3;
            
        case 'DMT'
            sub_list = 1:20;
            conditions = {'DMT', 'PCB'};
            runs = 1;
            
        case 'Anesthesia'
            sub_list = 1:16; % Customise based on your participants
            conditions = {'Awake', 'Unconscious', 'Recovery'};
            runs = 1;
            
        case 'Modafinil'
            sub_list = 1:19;
            conditions = {'MOD', 'PLB'};
            runs = 1;
            
        case 'Schizophrenia'
            sub_list = 1:20;
            conditions = {'SCHZ', 'CTRL'};
            runs = 1;
            
        otherwise
            error('Unknown dataset chosen.');
    end
    
    % Output Directory
    output_dir = fullfile(main_dir, 'results', 'func_networks', atlas_name, dataset);
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    %% ── PROCESSING LOOP ──────────────────────────────────────────────────────
    for iSess = 1:length(sub_list)
        sub_id = sprintf('sub-%03d', sub_list(iSess));
        
        for iCond = 1:length(conditions)
            cond_name = conditions{iCond};
            
            for iRun = runs
                % Define directories where raw NIfTI BOLD files are expected
                % (Change this path according to your local raw data directory)
                raw_func_dir = fullfile(main_dir, 'data', 'raw', sub_id, ['ses-', cond_name], 'func');
                
                % Standard BIDS naming convention
                if length(runs) > 1
                    nii_filename = sprintf('%s_ses-%s_task-rest_run-%02d_bold.nii.gz', sub_id, cond_name, iRun);
                else
                    nii_filename = sprintf('%s_ses-%s_task-rest_bold.nii.gz', sub_id, cond_name);
                end
                
                nii_filepath = fullfile(raw_func_dir, nii_filename);
                
                % In a standard reproducible repo, if raw files are not found,
                % we output a warning since raw fMRI images are usually stored externally.
                if ~exist(nii_filepath, 'file')
                    fprintf('[SKIP] Raw fMRI file not found for %s, %s, Run %d\n', sub_id, cond_name, iRun);
                    continue;
                end
                
                fprintf('Extracting BOLD for %s, Condition: %s, Run: %d...\n', sub_id, cond_name, iRun);
                
                % Load BOLD image
                bold_hdr = spm_vol(nii_filepath);
                bold_data = spm_read_vols(bold_hdr);
                
                % Extract regional time-series (bold_to_networks is in the same folder)
                func_roi = bold_to_networks(bold_data, mask_data);
                
                % Save output .mat
                out_sub_dir = fullfile(output_dir, sub_id, cond_name);
                if ~exist(out_sub_dir, 'dir')
                    mkdir(out_sub_dir);
                end
                
                if length(runs) > 1
                    out_filename = sprintf('%s_ses-%s_task-rest_run-%02d_bold_%s.mat', sub_id, cond_name, iRun, atlas_name);
                else
                    out_filename = sprintf('%s_ses-%s_task-rest_bold_%s.mat', sub_id, cond_name, atlas_name);
                end
                
                save(fullfile(out_sub_dir, out_filename), 'func_roi');
            end
        end
    end
    fprintf('=== Parcellation Pipeline Finished ===\n');
end
