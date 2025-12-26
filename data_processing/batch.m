function batch()

    config = jsondecode(fileread("Lab on a Mat\config.json"));
    file_folder_path = config.file_folder_path;

    csv_files = dir(fullfile(file_folder_path, '*.csv'));
    fprintf('找到 %d 个.csv文件\n', length(csv_files));

    for i = 1:length(csv_files)
        [~, filename, ~] = fileparts(csv_files(i).name);

        fprintf('正在处理第 %d/%d 个文件: %s\n', i, length(csv_files),  filename);

        try
            ret = process(filename);
            resistance(ret.group) = ret.resistance;
            slope(ret.group, ret.number) = ret.slope;
            slope_std_error(ret.group, ret.number) = ret.slope_std_error;
            fprintf('完成处理文件: %s\n\n', filename);
        catch ME
            fprintf('处理文件 %s 时出错: %s\n', filename, ME.message);
        end
    end

    fprintf('批量处理完成\n');

    capacitance = zeros(5, 1);
    slope_group = zeros(5, 1);
    % capacitance_rel_error = zeros(5, 1);
    capacitance_error = 0;
    for i = 1:5
        for j = 1:3
            slope_group(i) = slope_group(i) + slope(i, j) / 3;
        end
        capacitance(i) = 1 / (slope_group(i) * resistance(i));
        % capacitance_rel_error(i) = sqrt(power(slope_std_error(i) / (sqrt(3) * slope_group(i)), 2) + power(0.5 / resistance(i), 2));
    end

    capacitance_avg = sum(capacitance) / 5;

    for i = 1:5
        capacitance_error = capacitance_error + power(capacitance(i) - capacitance_avg, 2) / 4;
    end
    capacitance_error = sqrt(capacitance_error);

    fprintf('C[] = %.8f, %.8f, %.8f, %.8f, %.8f\n', capacitance(1), capacitance(2), capacitance(3), capacitance(4), capacitance(5));
    fprintf('C = %.8f ± %.8f = %.6f ± %.6f\n', capacitance_avg, capacitance_error, capacitance_avg, capacitance_error);
    fprintf('%%ΔC = %.3f\n', 100 * capacitance_error / capacitance_avg);
end