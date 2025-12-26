function mean_val = avg(csv_path, graph)
	% bg 读取 CSV 中的电压波形，计算均值并可视化，返回均值 mean_val。

	% 参数默认
	if nargin == 0
		csv_path = "D:\WHBC\Physics IA\data\final_data\bg.csv";
		graph = false;
	end
	graph = logical(graph);

	% 读取数据
	data = readmatrix(csv_path);
	if isempty(data)
		error('读取的文件为空或无法解析：%s', csv_path);
	end

	% 选择波形列：优先使用第二列（如存在），否则使用第一列；
	% 如果 readmatrix 返回向量则直接使用。
	if isvector(data)
		waveform = data(:);
	else
		if size(data,2) >= 2
			waveform = data(:,2);
		else
			waveform = data(:,1);
		end
	end

	% 丢弃 NaN 值用于统计
	valid = ~isnan(waveform);
	if ~any(valid)
		error('波形数据没有有效的数值（全为 NaN）。');
	end

	mean_val = mean(waveform(valid));

	% 输出均值
	% fprintf('平均电压（bg） = %.6f\n', mean_val);

	% 绘图（如果需要）
	if graph
		figure;
		plot(1:length(waveform), waveform, '.-');
		xlabel('Sample Point');
		ylabel('V /V');
		title('V-against-t Diagram');
		grid on;
	end
end

