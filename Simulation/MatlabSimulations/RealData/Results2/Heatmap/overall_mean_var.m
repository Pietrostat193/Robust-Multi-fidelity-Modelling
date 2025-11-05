%% Time-averaged heatmaps from per-time CSV grids (mean + var + std)
dataDir = pwd;  % folder containing grid_w*_t*.csv
files = dir(fullfile(dataDir, "grid_w*_t*.csv"));
files = files(~[files.isdir]);
if isempty(files)
    error("No grid_w*_t*.csv found in %s", dataDir);
end
fprintf("Found %d files\n", numel(files));

% --- Establish grid from first file ---
T0 = readtable(fullfile(files(1).folder, files(1).name));
x_unique = unique(T0.x);
y_unique = unique(T0.y);
nx = numel(x_unique);
ny = numel(y_unique);

% Accumulators (NaN-safe)
sum_mu  = zeros(ny,nx);
sum_var = zeros(ny,nx);
sum_std = zeros(ny,nx);
count   = zeros(ny,nx);

hasStd = false; % <--- Define it here so it always exists

% --- Loop over all time files ---
for k = 1:numel(files)
    T = readtable(fullfile(files(k).folder, files(k).name));
    if any(strcmpi(T.Properties.VariableNames,'std'))
        hasStd = true;
    end

    [~, ix] = ismember(T.x, x_unique);
    [~, iy] = ismember(T.y, y_unique);
    lin = sub2ind([ny,nx], iy, ix);

    mu_grid  = nan(ny,nx);  mu_grid(lin)  = T.mean;
    var_grid = nan(ny,nx);  var_grid(lin) = T.('var');
    if hasStd
        std_grid = nan(ny,nx); std_grid(lin) = T.('std');
    end

    mask_mu = isfinite(mu_grid);
    sum_mu(mask_mu) = sum_mu(mask_mu) + mu_grid(mask_mu);
    count(mask_mu)  = count(mask_mu) + 1;

    mask_var = isfinite(var_grid);
    sum_var(mask_var) = sum_var(mask_var) + var_grid(mask_var);

    if hasStd
        mask_std = isfinite(std_grid);
        sum_std(mask_std) = sum_std(mask_std) + std_grid(mask_std);
    end
end

% --- Compute time averages ---
valid = count > 0;
mu_avg  = nan(ny,nx);
var_avg = nan(ny,nx);
mu_avg(valid)  = sum_mu(valid)  ./ count(valid);
var_avg(valid) = sum_var(valid) ./ count(valid);
std_avg = nan(ny,nx);
if hasStd
    std_avg(valid) = sum_std(valid) ./ count(valid);
end
std_from_varavg = sqrt(var_avg);

% --- Plot results ---
hMean = figure('Color','w','Name','Time-averaged mean');
imagesc(x_unique, y_unique, mu_avg);
set(gca,'YDir','normal'); axis image; grid on; box on;
colormap(parula);
cb = colorbar; cb.Label.String = 'Time-avg mean';
xlabel('x'); ylabel('y');
title('Global time average of predicted mean');

hVar = figure('Color','w','Name','Time-averaged variance');
imagesc(x_unique, y_unique, var_avg);
set(gca,'YDir','normal'); axis image; grid on; box on;
colormap(parula);
cb = colorbar; cb.Label.String = 'Time-avg variance';
xlabel('x'); ylabel('y');
title('Global time average of predicted variance');

if hasStd
    hStd = figure('Color','w','Name','Time-averaged std (reported)');
    imagesc(x_unique, y_unique, std_avg);
    set(gca,'YDir','normal'); axis image; grid on; box on;
    colormap(parula);
    cb = colorbar; cb.Label.String = 'Time-avg std (reported)';
    xlabel('x'); ylabel('y');
    title('Global time average of reported std');
end

% --- Save outputs ---
doSave = true;
if doSave
    [Xg, Yg] = meshgrid(x_unique, y_unique);
    if hasStd
        Tavg = table(Xg(:), Yg(:), mu_avg(:), var_avg(:), std_avg(:), std_from_varavg(:), ...
                     'VariableNames', {'x','y','mean_avg','var_avg','std_avg','std_from_varavg'});
    else
        Tavg = table(Xg(:), Yg(:), mu_avg(:), var_avg(:), std_from_varavg(:), ...
                     'VariableNames', {'x','y','mean_avg','var_avg','std_from_varavg'});
    end
    writetable(Tavg, fullfile(dataDir,'timeavg_fields_grid.csv'));

    exportgraphics(hMean, fullfile(dataDir,'timeavg_mean_heatmap.png'), 'Resolution', 200);
    exportgraphics(hVar,  fullfile(dataDir,'timeavg_var_heatmap.png'),  'Resolution', 200);
    if hasStd
        exportgraphics(hStd, fullfile(dataDir,'timeavg_std_heatmap.png'), 'Resolution', 200);
    end
end
