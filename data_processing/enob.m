function enob(filepath, n)
    % 读取CSV文件
    data = readmatrix(filepath);
    
    % 提取第二列第三行及以下的数据作为波形数据
    waveform = data(3:end, 2);
    
    % 计算LSB（任意两个值之差的最小值）
    sorted_wave = sort(waveform);
    diffs = diff(sorted_wave);
    lsb = min(diffs(diffs > 0));  % 取大于0的最小差值
    
    % 计算最小值和最大值
    min_val = min(waveform);
    max_val = max(waveform);
    
    % 计算n分位和(1-n)分位的值
    n_percentile = prctile(waveform, n * 100);
    one_minus_n_percentile = prctile(waveform, (1 - n) * 100);
    
    % 找到n分位值最后一次出现的横坐标
    n_indices = find(abs(waveform - n_percentile) < eps);
    n_last_index = n_indices(end);
    
    % 找到(1-n)分位值最后一次出现的横坐标
    one_minus_n_indices = find(abs(waveform - one_minus_n_percentile) < eps);
    one_minus_n_last_index = one_minus_n_indices(end);
    
    % 在控制台打印参数
    fprintf('LSB: %.6f\n', lsb);
    fprintf('最小值: %.6f\n', min_val);
    fprintf('最大值: %.6f\n', max_val);
    % fprintf('%g分位值: %.6f，第一次出现位置: %d\n', n, n_percentile, n_last_index);
    % fprintf('%g分位值: %.6f，最后一次出现位置: %d\n', 1-n, one_minus_n_percentile, one_minus_n_last_index);
    % fprintf('有效长度: %d\n', one_minus_n_last_index - n_last_index);
end