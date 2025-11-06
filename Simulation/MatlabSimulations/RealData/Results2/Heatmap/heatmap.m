%% 11) Heatmap media e incertezza sull'intero dominio di training (per ogni t della finestra)
% Requisiti:
%   - ModelInfo.hyp già impostato (p.es. agli ultimi iperparametri caricati)
%   - predict2Dsp supporta [mu, s2] = predict2Dsp(X) (se no, la mappa incertezza verrà ignorata)
%
% Output:
%   - Figura interattiva per ciascun time step (mean + variance nello stesso plot)
%   - Salvataggio CSV per ogni time step (x,y,mean,var,std)
%   - Log CSV con medie spaziali di mean e variance per time step

doSavePng   = false;                 % metti true per salvare i PNG
outDirHeat  = "heatmaps_window";     % cartella immagini
outDirData  = "heatmaps_csv";        % cartella CSV per griglie e summary
if doSavePng && ~exist(outDirHeat,'dir'); mkdir(outDirHeat); end
if ~exist(outDirData,'dir'); mkdir(outDirData); end
summaryCSV = fullfile(outDirData, 'summary_per_time.csv');
if ~isfile(summaryCSV)
    writetable(cell2table({'window','time','mean_mean','mean_var','mean_std'}, ...
        'VariableNames', {'window','time','mean_mean','mean_var','mean_std'}), summaryCSV);
    % Rimpiazza header con tipi numerici al primo append reale:
    delete(summaryCSV);
end

% Dominio spaziale = bounding box dei dati (LF o HF) nella finestra
XY_all = [LF_subtime(:,3:4); HF_subtime(:,3:4)];
minX = min(XY_all(:,1)); maxX = max(XY_all(:,1));
minY = min(XY_all(:,2)); maxY = max(XY_all(:,2));

% Piccolo margine estetico intorno al dominio
padX = 0.02*(maxX-minX + eps);  padY = 0.02*(maxY-minY + eps);
minX = minX - padX; maxX = maxX + padX;
minY = minY - padY; maxY = maxY + padY;

% Risoluzione della griglia (aumenta se vuoi mappe più “lisce”)
nx = 120; ny = 120;
xg = linspace(minX, maxX, nx);
yg = linspace(minY, maxY, ny);
[Xg, Yg] = meshgrid(xg, yg);

% Punti di training (per overlay): HF usati per il fit e HF test (site==1)
HF_tr_XY   = HF_selected(:,[3,4]);           % train HF
HF_tr_t    = HF_selected(:,2);
HF_test_XY = HF_subtime(HF_subtime(:,1)==1,[3,4]);  % test HF (site==1)
HF_test_t  = HF_subtime(HF_subtime(:,1)==1,2);

% Per uniformità di colori tra i tempi, opzionalmente calcola range globale
% su un subset di tempi (qui usiamo i min/max dinamici per semplicità).
linkColorAcrossTimes = false;

for it = 1:numel(keep_times)
    tcurr = keep_times(it);
disp(it)
    % Query su tutta la griglia a tempo tcurr
    Xquery = [tcurr*ones(numel(Xg),1), Xg(:), Yg(:)];

    mu_grid  = nan(size(Xg));
    var_grid = nan(size(Xg));   % varianza (se disponibile)
    try
        % Prova: media + varianza
        [mu_vec, s2_vec] = predict2Dsp2(Xquery);
        mu_grid(:)  = mu_vec;
        if ~isempty(s2_vec) && all(isfinite(s2_vec))
            var_grid(:) = max(s2_vec,0);
        end
    catch
        % Fallback: solo media
        try
            mu_vec = predict2Dsp2(Xquery);
            mu_grid(:) = mu_vec;
            warning('predict2Dsp non fornisce varianza: la mappa di incertezza sara'' omessa (t=%g).', tcurr);
        catch ME
            warning('Predizione fallita alla time=%g: %s', tcurr, ME.message);
            continue;
        end
    end
    std_grid = sqrt(var_grid);  % potrà contenere NaN se var non disponibile

    % Colormap limits (mean)
    if ~exist('cLimMu','var') || ~linkColorAcrossTimes
        muFinite = mu_grid(isfinite(mu_grid));
        if ~isempty(muFinite)
            cLimMu = [prctile(muFinite,2), prctile(muFinite,98)];
            if diff(cLimMu)<=0, cLimMu = [min(muFinite), max(muFinite)+eps]; end
        else
            cLimMu = [0 1];
        end
    end

    % Medie spaziali (conditional mean & variance)
    mean_mean = mean(mu_grid(isfinite(mu_grid)),'omitnan');
    mean_var  = mean(var_grid(isfinite(var_grid)),'omitnan');
    mean_std  = mean(std_grid(isfinite(std_grid)),'omitnan');

    % -------- Salvataggi CSV per questo time step --------
    % 1) Griglia completa
    Tgrid = table( Xg(:), Yg(:), mu_grid(:), var_grid(:), std_grid(:), ...
        'VariableNames', {'x','y','mean','var','std'} );
    csvName = fullfile(outDirData, sprintf('grid_w%03d_t%g.csv', w, tcurr));
    writetable(Tgrid, csvName);

    % 2) Riga summary per time step
    Tsum = table( w, tcurr, mean_mean, mean_var, mean_std, ...
        'VariableNames', {'window','time','mean_mean','mean_var','mean_std'} );
    if isfile(summaryCSV)
        writetable(Tsum, summaryCSV, 'WriteMode','append');
    else
        writetable(Tsum, summaryCSV);
    end

    % Seleziona i punti HF (train/test) a tcurr per overlay
    isTrNow   = (HF_tr_t   == tcurr);
    isTestNow = (HF_test_t == tcurr);
    trXY_now   = HF_tr_XY(isTrNow,:);
    testXY_now = HF_test_XY(isTestNow,:);

    % ----------------- Plot unico: mean + variance -----------------
    f = figure('Color','w','Name',sprintf('Mean+Var t=%g', tcurr), ...
               'Units','normalized','Position',[0.1 0.1 0.75 0.7]);
    ax = axes(f); hold(ax,'on'); grid(ax,'on'); box(ax,'on');

    % Mean come heatmap
    imagesc(ax, xg, yg, mu_grid); set(ax,'YDir','normal'); axis(ax,'image');
    colormap(ax, parula); caxis(ax, cLimMu);
    cb1 = colorbar(ax, 'Location','eastoutside'); cb1.Label.String = 'Media predetta';

    % Variance come isocontorni (se disponibile)
    if any(isfinite(var_grid(:)))
        % scegli un numero moderato di livelli
        vFinite = var_grid(isfinite(var_grid));
        vLevels = linspace(prctile(vFinite,10), prctile(vFinite,90), 6);
        [C,hc] = contour(ax, xg, yg, var_grid, vLevels, 'LineWidth', 1.1);
        clabel(C,hc,'Color',[0.1 0.1 0.1],'FontSize',9,'LabelSpacing',200);
        hc.LineColor = [0.1 0.1 0.1];
    else
        % Nessuna varianza: solo nota testuale
        text(ax, 0.02, 0.02, 'Varianza non disponibile', 'Units','normalized', ...
            'FontWeight','bold','BackgroundColor','w','Margin',4);
    end

    % Overlay punti HF
    if ~isempty(trXY_now)
        plot(ax, trXY_now(:,1),  trXY_now(:,2),  'k.', 'MarkerSize',10, 'DisplayName','HF train');
    end
    if ~isempty(testXY_now)
        plot(ax, testXY_now(:,1), testXY_now(:,2), 'wo', 'MarkerSize',6, 'LineWidth',1.2, 'DisplayName','HF test');
    end
    xlabel(ax,'x'); ylabel(ax,'y');
    title(ax, sprintf('Posterior mean (+ variance contours) | finestra %d/%d | t = %g', w, nFull, tcurr));
    legend(ax,'Location','southoutside','Orientation','horizontal','Box','off');

    % Salvataggio opzionale PNG
    if doSavePng
        pngName = fullfile(outDirHeat, sprintf('mean_var_w%03d_t%g.png', w, tcurr));
        exportgraphics(f, pngName, 'Resolution', 180);
    end
end
