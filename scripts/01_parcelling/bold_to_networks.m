function func_roi = bold_to_networks(data, mask_data)

num_roi = length(unique(mask_data)) - 1;
numbers_roi = 1:num_roi;
x_size = size(data, 1);
y_size = size(data, 2);
z_size = size(data, 3);
t_size = size(data, 4);

func_roi = zeros(num_roi, t_size);
for r=1:num_roi
    for t=1:t_size
        mascara_region = mask_data == numbers_roi(r);
        
        mask = reshape(mascara_region, x_size*y_size*z_size, 1);
        data_reshape = reshape(data(:,:,:,t), x_size*y_size*z_size, 1);
        func_roi(r,t) = mean(data_reshape(mask));
    end
end
end