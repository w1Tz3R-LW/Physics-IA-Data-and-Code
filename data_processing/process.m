function ret = process(filename)
    config = jsondecode(fileread("Lab on a Mat\config.json"));
    file_folder_path = config.file_folder_path;
    experiment_params = jsondecode(fileread(strcat(file_folder_path, 'experiment_params.json')));
    file_path = strcat(file_folder_path, filename, ".csv");
    n = config.n;
    m = config.m;
    k = config.k;
    should_graph_preprocess = config.should_graph_preprocess;
    should_graph_fitting = config.should_graph_fitting;
    preprocess_left_start_idx = config.preprocess_left_start_idx;
    preprocess_left_end_idx = config.preprocess_left_end_idx;
    V_0 = avg(config.V_path, false);
    group = experiment_params.(strcat('x', filename)).group;
    number = experiment_params.(strcat('x', filename)).number;
    resistance = experiment_params.(strcat('x', filename)).resistance;
    sampling_rate = experiment_params.(strcat('x', filename)).sampling_rate;

    data = readmatrix(file_path);
    
    % 创建两行多列的 waveform 数组
    % 第一行是时间点，第二行是波形数据
    % 将时间点除以采样率，得到以秒为单位的时间
    waveform = [data(3:end, 1)' / sampling_rate; data(3:end, 2)'];

    % graph([data(3:end, 1)'; data(3:end, 2)']);

    % 调用预处理函数
    waveform = preprocess(waveform, n, m, k, V_0, should_graph_preprocess, preprocess_left_start_idx, preprocess_left_end_idx);

    % graph(waveform);
    
    fit_result = fitting(waveform, V_0, should_graph_fitting);

    % 构建返回结构体
    ret.group = group;
    ret.number = number;
    ret.resistance = resistance;
    ret.waveform = fit_result.waveform;
    ret.slope = fit_result.slope;
    ret.slope_std_error = fit_result.slope_std_error;

end

function graph(waveform)
    % GRAPH 绘制波形图
    % 输入：
    %   waveform - 需要绘图的波形数据 (2行N列，第1行是时间点，第2行是波形数据)

    % 创建新的图形窗口
    figure;

    % 绘制散点图，不连接每个数据点
    scatter(waveform(1, :), waveform(2, :), 10, 'b', 'filled');

    % 添加标题和标签
    title('V-against-t Graph', 'FontSize', 14);
    xlabel('Time (s)', 'FontSize', 12);
    ylabel('V /V', 'FontSize', 12);

    % 添加网格
    grid on;

    % 优化显示
    box on;
end

function ret = preprocess(waveform, n_percent, m, k, V_0, should_graph_preprocess, preprocess_left_start_idx, preprocess_left_end_idx)
    % PREPROCESS 预处理波形数据
    % 输入：
    %   waveform - 未处理的波形数组 (2行N列，第1行是时间点，第2行是波形数据)
    %   n_percent - 电压比例参数
    %   m - 平均值计算的点数间隔
    %   k - 连续超过阈值的点数
    %   V_0 - 电源电压值
    %   should_graph_preprocess - 是否绘制预处理图像
    %   preprocess_left_start_idx - 计算阈值的起始索引（从后往前数）
    %   preprocess_left_end_idx - 计算阈值的结束索引（从后往前数）
    % 输出：
    %   ret - 处理后的波形数组

    % 提取时间点和波形数据
    time_points = waveform(1, :);
    wave_data = waveform(2, :);

    % 每m个点取平均值（非重叠分组方式）
    len = length(wave_data);
    num_groups = floor(len / m);

    % 如果有足够的点进行分组平均
    if num_groups > 0
        % 初始化结果数组为行向量
        time_points_avg = zeros(1, num_groups);
        waveform_avg = zeros(1, num_groups);

        % 对每个完整的m点组计算平均值
        for i = 1:num_groups
            start_idx_avg = (i-1) * m + 1;
            end_idx_avg = i * m;
            time_points_avg(i) = mean(time_points(start_idx_avg:end_idx_avg));
            waveform_avg(i) = mean(wave_data(start_idx_avg:end_idx_avg));
        end
    else
        % 如果没有足够的点，则返回原始数据
        time_points_avg = time_points;
        waveform_avg = wave_data;
    end

    % graph([time_points_avg; waveform_avg]);

    % 找到V_0 * n_percent电压值第一次出现的位置，并截断
    threshold_voltage = V_0 * n_percent;
    first_index = find(waveform_avg >= threshold_voltage, 1, 'first');

    % disp(first_index);

    % 如果找到了这样的位置，且该位置不是太靠后，则截断数组
    if ~isempty(first_index)
        time_points_avg = time_points_avg(1:first_index);
        waveform_avg = waveform_avg(1:first_index);
    end

    % 保存去除前端之前的数据用于绘图
    waveform_before_front_removal = waveform_avg;
    time_before_front_removal = time_points_avg;  % 保存对应的时间点
    
    % 对于差分数据，我们使用时间点的中间值作为横坐标
    if length(time_before_front_removal) > 1
        time_diff = (time_before_front_removal(1:end-1) + time_before_front_removal(2:end)) / 2;
    else
        time_diff = [];
    end

    % 计算相邻点的差分（近似导数）
    if length(waveform_avg) > 1
        diff_waveform = diff(waveform_avg);
    else
        diff_waveform = waveform_avg;
    end

    % 计算指定范围内差分值的平均值
    % 从后往前数，计算第preprocess_left_start_idx个到第preprocess_left_end_idx个差分值的平均值
    if length(diff_waveform) >= preprocess_left_end_idx
        % 确定从后往前数的索引范围
        reverse_start_idx = length(diff_waveform) - preprocess_left_start_idx + 1;
        reverse_end_idx = length(diff_waveform) - preprocess_left_end_idx + 1;
        
        % 确保索引在有效范围内
        actual_start_idx = max(1, reverse_end_idx);
        actual_end_idx = min(length(diff_waveform), reverse_start_idx);
        
        threshold = mean(diff_waveform(actual_start_idx:actual_end_idx));

        % 找到从该点开始往后的连续k个点都超过阈值的位置
        first_exceed_index = NaN;
        for i = 1:(length(diff_waveform) - k + 1)  % 确保有足够多的点进行检查
            % 检查从当前位置开始的连续k个点是否都超过阈值
            if all(diff_waveform(i:i+k-1) > threshold)
                first_exceed_index = i;
                break;
            end
        end

        % 如果找到了符合条件的位置，则去除该位置以前的数据
        % 注意：first_exceed_index是diff_waveform中的索引，需要转换为waveform_avg中的索引
        if ~isnan(first_exceed_index)
            % 由于diff_waveform是waveform_avg的差分，索引可以直接使用
            % 但需要确保不超过waveform_avg的长度
            if first_exceed_index <= length(waveform_avg)
                time_points_avg = time_points_avg(first_exceed_index:end);
                waveform_avg = waveform_avg(first_exceed_index:end);
            end
        end
        
        % 将时间值重新调整，使第一个数据点的时间为0
        if ~isempty(time_points_avg)
            time_points_avg = time_points_avg - time_points_avg(1);
        end
    else
        threshold = NaN;
        first_exceed_index = NaN;
    end

    if should_graph_preprocess
        figure;
        % display_range = round(4 * first_exceed_index);
        display_range = length(diff_waveform);

        % 第一个子图：原始波形
        subplot(2, 1, 1);
        x_range1 = time_before_front_removal(1:min(length(waveform_before_front_removal), display_range));
        y_range1 = waveform_before_front_removal(1:min(length(waveform_before_front_removal), display_range));
        scatter(x_range1, y_range1, 10, 'b', 'filled');
        xlim([min(x_range1), max(x_range1)]);
        title('Waveform after tail removal (before front removal)', 'FontSize', 12);
        xlabel('Time (s)', 'FontSize', 10);
        ylabel('Voltage', 'FontSize', 10);
        grid on;

        if ~isnan(first_exceed_index) && first_exceed_index <= display_range
            % 使用实际的时间值作为cut point
            if first_exceed_index <= length(time_before_front_removal)
                cut_time = time_before_front_removal(first_exceed_index);
                hold on;
                xline(cut_time, 'r--', 'LineWidth', 1, 'Label', 'Front cut point');
                legend('Waveform', 'Front cut point', 'Location', 'northeast');
                hold off;
            else
                legend('Waveform', 'Location', 'northeast');
            end
        else
            legend('Waveform', 'Location', 'northeast');
        end

        % 第二个子图：差分
        subplot(2, 1, 2);
        % 对于差分数据，我们使用时间点的中间值作为横坐标
        if length(time_before_front_removal) > 1 && ~isempty(time_diff)
            x_range2 = time_diff(1:min(length(diff_waveform), display_range));
        else
            x_range2 = 1:min(length(diff_waveform), display_range);
        end
        y_range2 = diff_waveform(1:min(length(diff_waveform), display_range));
        scatter(x_range2, y_range2, 10, 'r', 'filled');
        xlim([min(x_range2), max(x_range2)]);
        title('First Difference (Approximate Derivative)', 'FontSize', 12);
        xlabel('Time (s)', 'FontSize', 10);
        ylabel('Difference', 'FontSize', 10);
        grid on;

        % 添加阈值横线
        if ~isnan(threshold)
            hold on;
            yline(threshold, 'k--', 'LineWidth', 1, 'Label', 'Threshold');
            if ~isnan(first_exceed_index) && first_exceed_index <= display_range
                % 使用实际的时间值作为cut point
                if first_exceed_index <= length(time_diff)
                    cut_time_diff = time_diff(first_exceed_index);
                    xline(cut_time_diff, 'r--', 'LineWidth', 1, 'Label', 'Front cut point', 'LabelVerticalAlignment', 'bottom');
                end
                legend('Difference', 'Threshold', 'Front cut point', 'Location', 'northeast');
            else
                legend('Difference', 'Threshold', 'Location', 'northeast');
            end
            hold off;
        else
            if ~isnan(first_exceed_index) && first_exceed_index <= display_range
                % 使用实际的时间值作为cut point
                hold on;
                if first_exceed_index <= length(time_diff)
                    cut_time_diff = time_diff(first_exceed_index);
                    xline(cut_time_diff, 'r--', 'LineWidth', 1, 'Label', 'Front cut point', 'LabelVerticalAlignment', 'bottom');
                end
                legend('Difference', 'Front cut point', 'Location', 'northeast');
                hold off;
            else
                legend('Difference', 'Location', 'northeast');
            end
        end

        % 添加区间标记: 从后往前数，标记preprocess_left_start_idx到preprocess_left_end_idx的区间
        if length(diff_waveform) >= preprocess_left_end_idx
            % 确定从后往前数的索引范围
            reverse_start_idx = length(diff_waveform) - preprocess_left_start_idx + 1;
            reverse_end_idx = length(diff_waveform) - preprocess_left_end_idx + 1;
            
            % 确保索引在有效范围内
            actual_start_idx = max(1, reverse_end_idx);
            actual_end_idx = min(length(diff_waveform), reverse_start_idx);
            
            % 限制在显示范围内
            actual_start_idx = min(actual_start_idx, display_range);
            actual_end_idx = min(actual_end_idx, display_range);
            
            % 获取y轴范围
            y_limits = ylim();
            
            % 添加灰色区域标记（使用实际时间值）
            if length(time_diff) >= actual_end_idx
                hold on;
                x_start = time_diff(actual_start_idx);
                x_end = time_diff(actual_end_idx);
                rectangle('Position', [x_start, y_limits(1), x_end-x_start, y_limits(2)-y_limits(1)], ...
                          'FaceAlpha', 0.2, 'EdgeColor', 'none', 'FaceColor', '#666666');
                hold off;
            end
        end
    end

    % 删除第一个点，让第二个点成为新的第一个点
    if length(waveform_avg) > 1
        time_points_final = time_points_avg(2:end);
        waveform_final = waveform_avg(2:end);
    else
        time_points_final = time_points_avg;
        waveform_final = waveform_avg;
    end
    
    % 返回两行多列的数组，第一行是时间点，第二行是波形数据
    ret = [time_points_final; waveform_final];
end

function ret = fitting(waveform, V_0, should_graph_fitting)
    % FITTING 对波形数据进行对数变换和线性拟合
    % 输入：
    %   waveform - 预处理后的波形数组 (2行N列，第1行是时间点，第2行是波形数据)
    %   V_0 - 电源电压值
    %   should_graph_fitting - 是否绘制拟合图像
    % 输出：
    %   ret - 包含拟合结果的结构体
    %         .waveform - 拟合后的曲线数据
    %         .slope - 拟合斜率
    %         .intercept - 拟合截距
    %         .slope_std_error - 斜率的标准误差
    %         .rmse - 均方根误差
    %         .r_squared - 决定系数

    % 检查输入数据的有效性
    if isempty(waveform) || size(waveform, 2) == 0
        ret.waveform = waveform;
        ret.slope = NaN;
        ret.intercept = NaN;
        ret.slope_std_error = NaN;
        ret.rmse = NaN;
        ret.r_squared = NaN;
        return;
    end

    % 提取时间点和波形数据
    t = waveform(1, :);
    wave_data = waveform(2, :);

    % 确保V_0不为零且大于波形中的最大值
    max_V = max(wave_data);
    if V_0 <= max_V
        warning('V_0 (%f) 应该大于波形中的最大值 (%f)，使用默认值进行计算', V_0, max_V);
        V_0 = max_V * 1.1;  % 使用略大于最大值的值
    end

    % 避免V接近V_0导致分母为零或负数
    epsilon = 1e-10;
    valid_indices = wave_data < (V_0 - epsilon);
    if sum(valid_indices) == 0
        warning('没有有效的数据点可用于拟合');
        ret.waveform = waveform;
        ret.slope = NaN;
        ret.intercept = NaN;
        ret.slope_std_error = NaN;
        ret.rmse = NaN;
        ret.r_squared = NaN;
        return;
    end

    % 只使用有效的数据点
    valid_t = t(valid_indices);
    valid_waveform = wave_data(valid_indices);

    % 计算变换后的y值: y = ln(V_0 / (V_0 - V))
    y_transformed = log(V_0 ./ (V_0 - valid_waveform));

    % 进行线性拟合: y = a*t + b
    % 使用polyfit获得系数和误差估计
    [coefficients, S] = polyfit(valid_t, y_transformed, 1);
    a = coefficients(1);  % 斜率
    b = coefficients(2);  % 截距

    % 计算斜率的标准误差
    % 根据MATLAB文档，S结构包含R和df信息用于计算标准误差
    % std_err = sqrt(diag(inv(R'*R))*mse) 其中 mse = norm(r)^2/df
    if isfield(S, 'R') && isfield(S, 'df')
        R = S.R;
        df = S.df;
        residuals = y_transformed - polyval(coefficients, valid_t);
        mse = sum(residuals.^2) / df;  % 均方误差
        % 计算协方差矩阵的对角线元素
        cov_matrix = inv(R' * R) * mse;
        % 斜率的标准误差是协方差矩阵第一个对角元素的平方根
        slope_std_error = sqrt(cov_matrix(1, 1));
    else
        % 如果无法计算标准误差，则设为NaN
        slope_std_error = NaN;
    end

    % 计算拟合值
    y_fitted = polyval(coefficients, valid_t);
    
    % 计算 RMSE
    rmse = sqrt(mean((y_transformed - y_fitted).^2));
    % 计算 R^2
    ss_res = sum((y_transformed - y_fitted).^2);      % 残差平方和
    ss_tot = sum((y_transformed - mean(y_transformed)).^2);    % 总平方和
    r_squared = 1 - (ss_res / ss_tot);

    % 在控制台打印拟合结果
    fprintf('线性拟合结果:\n');
    % fprintf('斜率: %.6f\n', a);
    % fprintf('截距: %.6f\n', b);
    fprintf('拟合方程: y = %.8f * t + %.8f\n', a, b);
    fprintf('RMSE: %.8f\n', rmse);
    fprintf('R^2: %.6f\n', r_squared);
    if ~isnan(slope_std_error)
        fprintf('斜率标准误差: %.8f\n', slope_std_error);
    else
        fprintf('斜率标准误差: 无法计算\n');
    end

    % 如果需要绘图
    if nargin < 3 || should_graph_fitting
        figure;
        scatter(valid_t, y_transformed, 10, 'b', 'filled');
        hold on;
        plot(valid_t, y_fitted, 'r-', 'LineWidth', 1.5);
        xlabel('Time (s)');
        ylabel('Transformed y-values');
        title(sprintf('y-t Graph and fitting result (y = %.4f*t + %.4f)', a, b));
        legend('y-values', 'fitting result', 'Location', 'best');
        % xlim([0.3, 0.4]);
        grid on;
        box on;
        hold off;
    end

    % 返回结构体结果
    ret.waveform = [valid_t; valid_waveform];
    ret.slope = a;
    ret.intercept = b;
    ret.slope_std_error = slope_std_error;
    ret.rmse = rmse;
    ret.r_squared = r_squared;
end