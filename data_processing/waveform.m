function waveform(csv_path, n, graph)

    % 读取CSV文件并提取波形数据
    data = readmatrix(csv_path);
    waveform_data = data(3:end, 2);  % 提取第二列第三行及以后的数据

    % 处理默认参数: 如果未提供第三个参数或为空，默认绘图为关闭
    if nargin < 3 || isempty(graph)
        graph = false;
    end
    graph = logical(graph);
    
    % 数据预处理：若 n==0 则特判为使用全部数据，否则排序去重并找到临界值
    if exist('n','var') && isequal(n,0)
        % n==0: 使用整个数据区间
        lower_threshold = NaN;
        upper_threshold = NaN;
        last_lower_pos = 1;
        first_upper_pos = length(waveform_data);
    else
        sorted_unique = unique(waveform_data);
        num_points = length(sorted_unique);

        % 计算n%对应的索引
        lower_idx = ceil(n * num_points);
        upper_idx = floor((1-n) * num_points);

        % 处理边界情况
        if lower_idx >= upper_idx
            lower_idx = 1;
            upper_idx = num_points;
        end

        % 获取临界值
        lower_threshold = sorted_unique(lower_idx);
        upper_threshold = sorted_unique(upper_idx);

        % 找到临界值在原波形中的位置
        lower_positions = find(waveform_data <= lower_threshold);
        upper_positions = find(waveform_data >= upper_threshold);

        % 获取保留区间的范围
        if ~isempty(lower_positions)
            last_lower_pos = lower_positions(end);
        else
            last_lower_pos = 1;
        end

        if ~isempty(upper_positions)
            first_upper_pos = upper_positions(1);
        else
            first_upper_pos = length(waveform_data);
        end
    end

    % 输出分位阈值与对应横坐标到控制台（n==0 时特殊提示）
    if exist('n','var') && isequal(n,0)
        fprintf('n=0：使用全部数据点，起始位置 = %d，结束位置 = %d\n', last_lower_pos, first_upper_pos);
    else
        fprintf('%d%%分位阈值 = %.6f，最后一次出现位置 = %d\n', n*100, lower_threshold, last_lower_pos);
        fprintf('%d%%分位阈值 = %.6f，第一次出现位置 = %d\n', (1-n)*100, upper_threshold, first_upper_pos);
    end
    
    % 创建预处理后的数据（保留区间内的值，其余置为空）
    processed_waveform = NaN(size(waveform_data));
    processed_waveform(last_lower_pos:first_upper_pos) = ...
        waveform_data(last_lower_pos:first_upper_pos);
    
    % 准备拟合数据
    valid_indices = find(~isnan(processed_waveform));
    x_data = valid_indices;
    y_data = processed_waveform(valid_indices);
    
    % 指数衰减拟合 y = A * exp(-k * x) + C
    % 将 x_data 平移为从 1 开始以便拟合稳定
    if isempty(x_data) || length(x_data) < 3
        % 数据不足以进行稳健拟合
        A = NaN; k = NaN; C = NaN;
        y_fitted = NaN(size(x_data));
        rmse = NaN;
        r_squared = NaN;
    else
        x_fit = double(x_data - x_data(1) + 1);
        y_fit = double(y_data);

        % 初始猜测：A 为振幅，C 为末值，k 为 1/平均索引
        A0 = max(y_fit) - min(y_fit);
        C0 = min(y_fit);
        k0 = 1 / max(1, mean(x_fit));
        param0 = [A0, k0, C0];

        % 目标函数：最小化平方残差
        obj = @(p) sum((y_fit - (p(1) * exp(-p(2) * x_fit) + p(3))).^2);

        % 使用 fminsearch 优化（不依赖额外工具箱）
        opts = optimset('Display','off','MaxIter',2000,'TolX',1e-8,'TolFun',1e-8);
        try
            p_opt = fminsearch(obj, param0, opts);
        catch
            % 若 fminsearch 不可用，退回到线性对数拟合近似
            log_x = log(x_data);
            coefficients = polyfit(log_x, y_data, 1);
            % 将线性拟合近似转换回指数参数（仅作近似展示）
            a_lin = coefficients(1); b_lin = coefficients(2);
            A = NaN; k = NaN; C = NaN;
            y_fitted = a_lin * log_x + b_lin;
            residuals = y_data - y_fitted;
            rmse = sqrt(mean(residuals.^2));
            r_squared = 1 - sum(residuals.^2) / sum((y_data - mean(y_data)).^2);
            p_opt = [];
        end

        if ~isempty(p_opt)
            A = p_opt(1);
            k = p_opt(2);
            C = p_opt(3);
            y_fitted = A * exp(-k * x_fit) + C;
            residuals = y_fit - y_fitted;
            rmse = sqrt(mean(residuals.^2));
            r_squared = 1 - sum(residuals.^2) / sum((y_fit - mean(y_fit)).^2);
            % 将拟合值对应回原 x_data 横坐标
            % (绘图时使用 x_data 与 y_fitted 相同长度)
        end
    end
    
    % 绘图（仅当 graph 为 true 时）
    if graph
        figure;
        hold on;
        if exist('n','var') && isequal(n,0)
            % n==0 时：画出所有原始数据点，但不绘制拟合曲线
            plot(1:length(waveform_data), waveform_data, 'b.', 'MarkerSize', 8);
        else
            % 正常情况：绘制预处理后波形（中间区间）与拟合曲线
            plot(1:length(waveform_data), processed_waveform, 'b.', 'MarkerSize', 8);
            plot(x_data, y_fitted, 'r-', 'LineWidth', 2);
        end
        xlabel('时间点');
        ylabel('电压值');
        title(sprintf('波形数据与指数拟合 (A=%.4f, k=%.6f, C=%.4f, RMSE=%.6f, R²=%.4f)', ...
            A, k, C, rmse, r_squared));
        if exist('n','var') && isequal(n,0)
            legend('全部波形点', 'Location', 'best');
        else
            legend('预处理后波形', '对数拟合曲线', 'Location', 'best');
        end
        grid on;
        hold off;
    end
    
    % 输出拟合结果
    fprintf('拟合结果:\n');
    fprintf('  公式: y = %.6f * exp(-%.6f * x) + %.6f\n', A, k, C);
    fprintf('  均方根误差(RMSE): %.8f\n', rmse);
    fprintf('  决定系数(R²): %.6f\n', r_squared);
end