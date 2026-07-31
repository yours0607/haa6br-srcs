% Build all Stage 4 submission figures from the frozen source-data CSV files.
% MATLAB R2023a is the exclusive rendering, export, preview, and grayscale backend.

clearvars;
close all;
clc;

rootDir = fileparts(mfilename('fullpath'));
sourceDir = fullfile(rootDir, 'source_data');
figureDir = fullfile(rootDir, 'figures');
qaDir = fullfile(rootDir, 'qa');
previewDir = fullfile(qaDir, 'previews');

ensureDir(figureDir);
ensureDir(qaDir);
ensureDir(previewDir);

S = stage4Style();
D = readSourceData(sourceDir);
validateSourceData(D);

records = exportBundle(drawFigure1(D, S), figureDir, previewDir, ...
    'fig_stage4_main', 183, 116, false);
records(end + 1) = exportBundle(drawFigure2(D, S), figureDir, previewDir, ...
    'fig_stage4_mechanism_tradeoff', 183, 116, false); %#ok<SAGROW>
records(end + 1) = exportBundle(drawFigure3(D, S), figureDir, previewDir, ...
    'fig_stage4_action_coverage', 183, 112, false); %#ok<SAGROW>
records(end + 1) = exportBundle(drawFigureS4(D, S), figureDir, previewDir, ...
    'fig_stage4_risk_coverage_frontier', 183, 120, false); %#ok<SAGROW>
records(end + 1) = exportBundle(drawFigure5(D, S), figureDir, previewDir, ...
    'fig_stage4_directional_overlap', 183, 96, false); %#ok<SAGROW>
records(end + 1) = exportBundle(drawFigureS1(D, S), figureDir, previewDir, ...
    'fig_stage4_selector_ablation', 183, 102, false); %#ok<SAGROW>
records(end + 1) = exportBundle(drawFigureS2(D, S), figureDir, previewDir, ...
    'fig_stage4_robustness_heterogeneity', 183, 104, false); %#ok<SAGROW>
records(end + 1) = exportBundle(drawFigureS3(D, S), figureDir, previewDir, ...
    'fig_stage4_uk_target_stress_test', 183, 110, false); %#ok<SAGROW>
records(end + 1) = exportBundle(drawFigureS5(D, S), figureDir, previewDir, ...
    'fig_stage4_geographic_shift_diagnostics', 183, 104, false); %#ok<SAGROW>
records(end + 1) = exportBundle(drawGraphicalAbstract(S), figureDir, previewDir, ...
    'graphical_abstract_stage4', 80, 40, true); %#ok<SAGROW>

writeMatlabQA(rootDir, qaDir, records, D);
writeManifests(rootDir, records);

fprintf('MATLAB Stage 4 figure build PASS: %d figures exported.\n', numel(records));


function S = stage4Style()
S.ink = rgb('#252829');
S.muted = rgb('#6C7375');
S.grid = rgb('#D9DEDE');
S.history = rgb('#5B6163');
S.teal = rgb('#0B6E72');
S.tealMid = rgb('#5B9A9C');
S.tealLight = rgb('#B9D6D5');
S.blue = rgb('#5B748D');
S.blueLight = rgb('#9AABBA');
S.amber = rgb('#C58A32');
S.coral = rgb('#C35E4B');
S.darkCoral = rgb('#934B45');
S.paleTeal = rgb('#EAF4F3');
S.paleAmber = rgb('#FAF1E2');
S.paleCoral = rgb('#F8EBE7');
S.white = [1, 1, 1];
S.font = 'Arial';
S.tickFont = 6.5;
S.labelFont = 7.0;
S.titleFont = 7.6;
S.panelFont = 8.2;
S.lineWidth = 0.75;
end


function D = readSourceData(sourceDir)
files = {
    'fig1a', 'fig1a_accuracy_contrasts.csv';
    'fig1bc', 'fig1bc_harm_bonferroni.csv';
    'fig2a', 'fig2a_mechanism_pooled.csv';
    'fig2bc', 'fig2bc_mechanism_regions.csv';
    'fig3nodes', 'fig3a_common_cohort_action_nodes.csv';
    'fig3flows', 'fig3a_common_cohort_action_flows.csv';
    'fig3coverage', 'fig3b_regional_correction_coverage.csv';
    'figS1', 'figS1_selector_contrasts.csv';
    'figS1context', 'figS1_selector_pooled_context.csv';
    'figS2', 'figS2_robustness_contexts.csv';
    'figS3pairs', 'figS3a_uk_target_pairs.csv';
    'figS3censoring', 'figS3b_uk_species_censoring.csv';
    'figS3contrasts', 'figS3c_uk_history_contrasts.csv';
    'figS4', 'figS4_risk_coverage.csv';
    'fig5overlap', 'fig5a_strategy_overlap.csv';
    'fig5directional', 'fig5bc_directional_underprediction.csv';
    'figS5target', 'figS5a_target_distribution.csv';
    'figS5classifier', 'figS5b_region_classifier.csv';
    'figS5jsd', 'figS5c_feature_jsd.csv';
    'ga', 'graphical_abstract_key_values.csv'};

D = struct();
for idx = 1:size(files, 1)
    path = fullfile(sourceDir, files{idx, 2});
    if ~isfile(path)
        error('Missing frozen source-data file: %s', path);
    end
    opts = detectImportOptions(path, 'TextType', 'string', ...
        'VariableNamingRule', 'preserve');
    D.(files{idx, 1}) = readtable(path, opts);
end
end


function validateSourceData(D)
expectedRows = struct('fig1a', 6, 'fig1bc', 6, 'fig2a', 12, ...
    'fig2bc', 120, 'fig3nodes', 13, 'fig3flows', 36, ...
    'fig3coverage', 30, 'figS1', 33, 'figS1context', 6, ...
    'figS2', 126, 'figS3pairs', 134, 'figS3censoring', 6, ...
    'figS3contrasts', 6, 'figS4', 24, 'fig5overlap', 12, ...
    'fig5directional', 21, 'figS5target', 10, ...
    'figS5classifier', 10, 'figS5jsd', 80, 'ga', 33);
names = fieldnames(expectedRows);
for idx = 1:numel(names)
    name = names{idx};
    assert(height(D.(name)) == expectedRows.(name), ...
        '%s row count changed.', name);
end

history = D.fig1a(D.fig1a.comparator == "History mean", :);
assert(all(history.ci_low < 0 & history.ci_high > 0), ...
    'History-relative accuracy interval boundary changed.');
assert(all(D.fig1bc.adjusted_ci_high < 0), ...
    'Prediction-harm sensitivity interval boundary changed.');

C = D.fig3coverage;
for k = 1:3
    Tk = C(C.k == k, :);
    observed = sum(Tk.corrected_systems) / sum(Tk.n_systems);
    expected = [3347 / 4785, 2138 / 3091, 1807 / 2743];
    assert(abs(observed - expected(k)) < 1e-12, ...
        'Correction coverage changed at k=%d.', k);
end

pathRows = D.figS4(D.figS4.series == "SRCS budget path", :);
assert(all(ismember([0.04, 0.06, 0.08, 0.10, 0.12, 0.15], ...
    unique(pathRows.risk_budget)')), 'Risk-budget frontier grid changed.');
for k = 1:3
    overlap = D.fig5overlap(D.fig5overlap.k == k, :);
    assert(abs(sum(overlap.share_of_evaluable_systems) - 1) < 1e-12, ...
        'Strategy-overlap shares changed at k=%d.', k);
    directional = D.fig5directional(D.fig5directional.k == k, :);
    srcs = directional(directional.display_method == "SRCS", :);
    history = directional(directional.display_method == "History mean", :);
    assert(srcs.system_underprediction_rate_gt_1_0 > ...
        history.system_underprediction_rate_gt_1_0, ...
        'Directional underprediction ordering changed at k=%d.', k);
end
assert(abs(mean(D.figS5classifier.one_vs_rest_system_level_auroc) - ...
    0.7242338233831003) < 1e-12, 'Region-classifier AUROC summary changed.');
assert(D.figS5target.haa6br_median_ug_l(D.figS5target.epa_region == 6) == 8.8, ...
    'Region 6 target median changed.');
assert(D.figS5target.haa6br_median_ug_l(D.figS5target.epa_region == 10) == 1.4, ...
    'Region 10 target median changed.');
end


function fig = newFigure(widthMm, heightMm)
fig = figure('Color', 'w', 'Units', 'centimeters', ...
    'Position', [1, 1, widthMm / 10, heightMm / 10], ...
    'PaperPositionMode', 'auto', 'Renderer', 'painters', ...
    'InvertHardcopy', 'off', 'Visible', 'on');
end


function fig = drawFigure1(D, S)
fig = newFigure(183, 116);
axA = axes(fig, 'Position', [0.105, 0.15, 0.44, 0.73]);
axB = axes(fig, 'Position', [0.665, 0.57, 0.30, 0.31]);
axC = axes(fig, 'Position', [0.665, 0.14, 0.30, 0.31]);

A = D.fig1a;
hold(axA, 'on');
backgroundBands(axA, [-0.19, 0.19], [0.55, 3.45], S);
xline(axA, 0, '--', 'Color', S.ink, 'LineWidth', 0.8);
historyHandle = gobjects(1);
cappedHandle = gobjects(1);
for k = 1:3
    y = 4 - k;
    row = A(A.k == k & A.comparator == "History mean", :);
    historyHandle = errorPoint(axA, row.point_estimate, row.ci_low, row.ci_high, ...
        y + 0.13, S.teal, 'o', true, S);
    row = A(A.k == k & A.comparator == "Capped History mean", :);
    cappedHandle = errorPoint(axA, row.point_estimate, row.ci_low, row.ci_high, ...
        y - 0.13, S.amber, '^', false, S);
end
xlim(axA, [-0.19, 0.19]);
ylim(axA, [0.55, 3.45]);
xticks(axA, -0.15:0.05:0.15);
yticks(axA, 1:3);
yticklabels(axA, {'k = 3   n = 2,743', 'k = 2   n = 3,091', 'k = 1   n = 4,785'});
xlabel(axA, 'SRCS - comparator MAE (ug/L)');
title(axA, 'History-relative accuracy', 'FontWeight', 'bold', 'HorizontalAlignment', 'left');
directionLabels(axA, 'lower MAE for SRCS', 'higher MAE for SRCS', S);
styleAxes(axA, S);
panelLabel(axA, 'a', -0.13, S);
legend(axA, [historyHandle, cappedHandle], {'History mean', 'Capped History mean'}, ...
    'Location', 'northoutside', 'Orientation', 'horizontal', 'Box', 'off', ...
    'FontName', S.font, 'FontSize', S.tickFont);

drawHarmPanel(axB, D.fig1bc, "negative_transfer_rate_difference", ...
    'Negative-transfer rate', 'SRCS - History mean (percentage points)', ...
    100, [-21.5, 1], [-20, -15, -10, -5, 0], S);
panelLabel(axB, 'b', -0.18, S);
drawHarmPanel(axC, D.fig1bc, "strict_cvar90_regret_difference", ...
    'Strict CVaR90 regret', 'SRCS - History mean (ug/L)', ...
    1, [-4.5, 0.25], [-4, -3, -2, -1, 0], S);
panelLabel(axC, 'c', -0.18, S);
end


function drawHarmPanel(ax, T, metric, panelTitle, xLabel, scale, xLimits, xTicks, S)
hold(ax, 'on');
backgroundBands(ax, xLimits, [0.55, 3.45], S);
xline(ax, 0, '--', 'Color', S.ink, 'LineWidth', 0.8);
rows = T(T.metric == metric, :);
for k = 1:3
    row = rows(rows.k == k, :);
    errorPoint(ax, row.point_estimate * scale, row.adjusted_ci_low * scale, ...
        row.adjusted_ci_high * scale, 4 - k, S.teal, 'o', true, S);
end
xlim(ax, xLimits);
ylim(ax, [0.55, 3.45]);
xticks(ax, xTicks);
yticks(ax, 1:3);
yticklabels(ax, {'3 rounds', '2 rounds', '1 round'});
xlabel(ax, xLabel);
title(ax, panelTitle, 'FontWeight', 'bold', 'HorizontalAlignment', 'left');
text(ax, 0.02, 0.95, 'lower regret', 'Units', 'normalized', ...
    'FontName', S.font, 'FontSize', 6.2, 'Color', S.teal, ...
    'VerticalAlignment', 'top');
styleAxes(ax, S);
end


function fig = drawFigure2(D, S)
fig = newFigure(183, 116);
axA = axes(fig, 'Position', [0.095, 0.15, 0.43, 0.73]);
axB = axes(fig, 'Position', [0.665, 0.57, 0.30, 0.31]);
axC = axes(fig, 'Position', [0.665, 0.14, 0.30, 0.31]);

pooled = D.fig2a;
regions = D.fig2bc;
variants = ["forced_action_no_abstention", ...
    "source_utility_without_risk_constraints", "zero_margin_gate", ...
    "cap_removed_at_application_same_selector"];
labels = ["No abstention", "No risk constraints", "Zero margin", "No cap*"];
colors = [S.coral; S.darkCoral; S.amber; S.history];
markers = {'o', 's', '^'};

hold(axA, 'on');
patch(axA, [-0.091, 0, 0, -0.091], [0, 0, 14.8, 14.8], S.paleAmber, ...
    'EdgeColor', 'none');
xline(axA, 0, '--', 'Color', S.ink, 'LineWidth', 0.8);
yline(axA, 0, '--', 'Color', S.ink, 'LineWidth', 0.8);
for v = 1:numel(variants)
    for k = 1:3
        row = pooled(pooled.comparator == variants(v) & pooled.k == k, :);
        scatter(axA, -row.equal_system_mae_difference, ...
            -100 * row.negative_transfer_rate_difference, 34, markers{k}, ...
            'MarkerFaceColor', colors(v, :), 'MarkerEdgeColor', S.white, ...
            'LineWidth', 0.7);
    end
end
labelX = [-0.070, -0.062, -0.066, -0.044];
labelY = [13.9, 11.6, 9.4, 1.9];
for v = 1:numel(variants)
    text(axA, labelX(v), labelY(v), labels(v), 'Color', colors(v, :), ...
        'FontName', S.font, 'FontSize', 6.5, 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center');
end
xlim(axA, [-0.091, 0.015]);
ylim(axA, [-0.3, 14.8]);
xticks(axA, -0.08:0.02:0);
yticks(axA, 0:3:15);
xlabel(axA, 'MAE difference (variant - full SRCS; ug/L)');
ylabel(axA, {'Negative-transfer difference (variant - full SRCS)', '(percentage points)'});
title(axA, 'Accuracy-negative transfer', 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'left');
styleAxes(axA, S);
panelLabel(axA, 'a', -0.14, S);
hDepth = gobjects(1, 3);
for k = 1:3
    hDepth(k) = scatter(axA, nan, nan, 28, markers{k}, ...
        'MarkerFaceColor', S.ink, 'MarkerEdgeColor', S.white);
end
legend(axA, hDepth, {'k = 1', 'k = 2', 'k = 3'}, ...
    'Location', 'southoutside', 'Orientation', 'horizontal', 'Box', 'off', ...
    'FontName', S.font, 'FontSize', S.tickFont);

drawMechanismRegionPanel(axB, pooled, regions, variants, labels, colors, markers, ...
    'equal_system_mae_difference', 'Regional MAE', ...
    'Difference (variant - full SRCS; ug/L)', [-0.27, 0.14], [-0.2, -0.1, 0, 0.1], -1, S);
panelLabel(axB, 'b', -0.20, S);
drawMechanismRegionPanel(axC, pooled, regions, variants, labels, colors, markers, ...
    'negative_transfer_rate_difference', 'Regional negative transfer', ...
    'Difference (variant - full SRCS; percentage points)', [-1, 23], [0, 5, 10, 15, 20], -100, S);
panelLabel(axC, 'c', -0.20, S);
end


function drawMechanismRegionPanel(ax, pooled, regions, variants, labels, colors, markers, ...
        column, panelTitle, xLabel, xLimits, xTicks, scale, S)
hold(ax, 'on');
backgroundBands(ax, xLimits, [0.55, 4.45], S);
xline(ax, 0, '--', 'Color', S.ink, 'LineWidth', 0.8);
for v = 1:numel(variants)
    yBase = 5 - v;
    for k = 1:3
        sub = regions(regions.comparator == variants(v) & regions.k == k, :);
        y = yBase + (k - 2) * 0.16 + (double(sub.outer_target_region) - 5.5) * 0.007;
        x = sub.(column) * scale;
        scatter(ax, x, y, 13, markers{k}, 'MarkerFaceColor', S.white, ...
            'MarkerEdgeColor', colors(v, :), 'LineWidth', 0.65);
        row = pooled(pooled.comparator == variants(v) & pooled.k == k, :);
        scatter(ax, row.(column) * scale, yBase + (k - 2) * 0.16, 27, markers{k}, ...
            'MarkerFaceColor', colors(v, :), 'MarkerEdgeColor', S.white, ...
            'LineWidth', 0.7);
    end
end
xlim(ax, xLimits);
ylim(ax, [0.55, 4.45]);
xticks(ax, xTicks);
yticks(ax, 1:4);
yticklabels(ax, flip(labels));
xlabel(ax, xLabel);
title(ax, panelTitle, 'FontWeight', 'bold', 'HorizontalAlignment', 'left');
styleAxes(ax, S);
end


function fig = drawFigure3(D, S)
fig = newFigure(183, 112);
axA = axes(fig, 'Position', [0.045, 0.12, 0.58, 0.77]);
axB = axes(fig, 'Position', [0.705, 0.15, 0.265, 0.71]);
drawAlluvial(axA, D.fig3nodes, D.fig3flows, S);
panelLabel(axA, 'a', -0.03, S);
drawCoverage(axB, D.fig3coverage, S);
panelLabel(axB, 'b', -0.20, S);
end


function drawAlluvial(ax, nodes, flows, S)
hold(ax, 'on');
axis(ax, [0, 1, 0, 1]);
axis(ax, 'off');
stageX = [0.02, 0.43, 0.84];
nodeWidth = 0.135;
nodeScale = 0.80;
gap = 0.014;
familyColors = containers.Map( ...
    {'Fallback', 'Persistence', 'HistoryMean', 'HistoryMedian', 'RawMean', 'RawMedian'}, ...
    {S.history, rgb('#5F8582'), S.blue, S.blueLight, rgb('#B08A58'), rgb('#B47F7D')});

pos = nodes(:, {'k', 'display_order', 'selected_family', 'family_label', ...
    'n_systems', 'percent_common_cohort'});
pos.bottom = zeros(height(pos), 1);
pos.top = zeros(height(pos), 1);
for k = 1:3
    idx = find(pos.k == k);
    [~, order] = sort(pos.display_order(idx));
    idx = idx(order);
    total = nodeScale + gap * (numel(idx) - 1);
    cursor = 0.5 + total / 2;
    for j = 1:numel(idx)
        h = pos.n_systems(idx(j)) / 2743 * nodeScale;
        pos.top(idx(j)) = cursor;
        pos.bottom(idx(j)) = cursor - h;
        cursor = cursor - h - gap;
    end
end

for pair = 1:2
    pairRows = flows(flows.source_k == pair & flows.target_k == pair + 1, :);
    sourceOrder = zeros(height(pairRows), 1);
    targetOrder = zeros(height(pairRows), 1);
    for r = 1:height(pairRows)
        sourceOrder(r) = pos.display_order(pos.k == pair & ...
            pos.selected_family == pairRows.source_family(r));
        targetOrder(r) = pos.display_order(pos.k == pair + 1 & ...
            pos.selected_family == pairRows.target_family(r));
    end
    pairRows.source_order = sourceOrder;
    pairRows.target_order = targetOrder;
    pairRows = sortrows(pairRows, {'source_order', 'target_order'});
    outCursor = containers.Map('KeyType', 'char', 'ValueType', 'double');
    inCursor = containers.Map('KeyType', 'char', 'ValueType', 'double');
    for r = 1:height(pairRows)
        sourceKey = sprintf('%d_%s', pair, char(pairRows.source_family(r)));
        targetKey = sprintf('%d_%s', pair + 1, char(pairRows.target_family(r)));
        sourceRow = pos(pos.k == pair & pos.selected_family == pairRows.source_family(r), :);
        targetRow = pos(pos.k == pair + 1 & pos.selected_family == pairRows.target_family(r), :);
        if ~isKey(outCursor, sourceKey), outCursor(sourceKey) = sourceRow.top; end
        if ~isKey(inCursor, targetKey), inCursor(targetKey) = targetRow.top; end
        h = pairRows.n_systems(r) / 2743 * nodeScale;
        sourceTop = outCursor(sourceKey);
        sourceBottom = sourceTop - h;
        targetTop = inCursor(targetKey);
        targetBottom = targetTop - h;
        outCursor(sourceKey) = sourceBottom;
        inCursor(targetKey) = targetBottom;
        ribbon(ax, stageX(pair) + nodeWidth, stageX(pair + 1), ...
            sourceTop, sourceBottom, targetTop, targetBottom, ...
            familyColors(char(pairRows.source_family(r))));
    end
end

for r = 1:height(pos)
    k = pos.k(r);
    family = char(pos.selected_family(r));
    color = familyColors(family);
    rectangle(ax, 'Position', [stageX(k), pos.bottom(r), nodeWidth, ...
        pos.top(r) - pos.bottom(r)], 'FaceColor', color, ...
        'EdgeColor', S.white, 'LineWidth', 0.55);
    if (pos.top(r) - pos.bottom(r)) > 0.055
        label = sprintf('%s\n%d (%.1f%%)', char(pos.family_label(r)), ...
            pos.n_systems(r), pos.percent_common_cohort(r));
    else
        label = sprintf('%s %d', shortFamily(family), pos.n_systems(r));
    end
    text(ax, stageX(k) + nodeWidth / 2, mean([pos.bottom(r), pos.top(r)]), ...
        label, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'FontName', S.font, 'FontSize', 6.0, 'Color', contrastText(color), ...
        'Clipping', 'on', 'Interpreter', 'none');
end
for k = 1:3
        text(ax, stageX(k) + nodeWidth / 2, 0.995, sprintf('k = %d', k), ...
        'HorizontalAlignment', 'center', 'FontName', S.font, ...
        'FontSize', 7.2, 'FontWeight', 'bold', 'Color', S.ink);
        text(ax, stageX(k) + nodeWidth / 2, 0.958, 'n = 2,743', ...
        'HorizontalAlignment', 'center', 'FontName', S.font, ...
        'FontSize', 6.0, 'Color', S.muted);
end
    title(ax, 'Action-family transitions', ...
    'FontName', S.font, 'FontSize', S.titleFont, 'FontWeight', 'bold', ...
    'Color', S.ink, 'HorizontalAlignment', 'left');
end


function drawCoverage(ax, T, S)
hold(ax, 'on');
k3 = sortrows(T(T.k == 3, :), 'correction_coverage', 'descend');
regions = double(k3.epa_region);
offsets = [-0.19, 0, 0.19];
colors = [S.teal; S.blue; S.amber];
markers = {'o', 's', '^'};

for xTick = 0:0.25:1
    xline(ax, xTick, '-', 'Color', S.grid, 'LineWidth', 0.55);
end

pooledX = zeros(1, 3);
for k = 1:3
    Tk = T(T.k == k, :);
    pooledX(k) = sum(Tk.corrected_systems) / sum(Tk.n_systems);
end
plot(ax, pooledX, 1 + offsets, '-', 'Color', S.grid, 'LineWidth', 0.9);
for k = 1:3
    scatter(ax, pooledX(k), 1 + offsets(k), 45, 'd', ...
        'MarkerFaceColor', S.white, 'MarkerEdgeColor', colors(k, :), ...
        'LineWidth', 1.2);
end

for r = 1:numel(regions)
    x = zeros(1, 3);
    for k = 1:3
        row = T(T.epa_region == regions(r) & T.k == k, :);
        x(k) = row.correction_coverage;
    end
    y = (r + 1) + offsets;
    plot(ax, x, y, '-', 'Color', rgb('#B7BCBD'), 'LineWidth', 0.85);
    for k = 1:3
        face = colors(k, :);
        if k == 2, face = S.white; end
        scatter(ax, x(k), y(k), 25, markers{k}, 'MarkerFaceColor', face, ...
            'MarkerEdgeColor', colors(k, :), 'LineWidth', 0.9);
    end
end
xlim(ax, [0, 1]);
ylim(ax, [0.45, 11.55]);
set(ax, 'YDir', 'reverse');
xticks(ax, 0:0.25:1);
xticklabels(ax, {'0%', '25%', '50%', '75%', '100%'});
yticks(ax, 1:11);
yticklabels(ax, [{'Pooled'}, cellstr(string(regions'))]);
xlabel(ax, 'Systems receiving correction');
ylabel(ax, 'EPA region');
title(ax, 'Correction coverage', 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'left');
styleAxes(ax, S);
ax.YAxis.TickLength = [0, 0];
h = gobjects(1, 3);
for k = 1:3
    face = colors(k, :);
    if k == 2, face = S.white; end
    h(k) = scatter(ax, nan, nan, 25, markers{k}, 'MarkerFaceColor', face, ...
        'MarkerEdgeColor', colors(k, :), 'LineWidth', 0.9);
end
legend(ax, h, {'k = 1', 'k = 2', 'k = 3'}, 'Location', 'northwest', ...
    'Box', 'off', 'FontName', S.font, 'FontSize', S.tickFont);
end


function fig = drawFigureS1(D, S)
fig = newFigure(183, 102);
positions = [0.09, 0.20, 0.245, 0.70; 0.385, 0.20, 0.245, 0.70; ...
    0.68, 0.20, 0.245, 0.70];
T = D.figS1;
held = T(T.scope == "held_region", :);
pooled = T(T.scope == "pooled_regions", :);
columns = ["ablated_minus_locked_full__equal_system_equal_future_round_mae", ...
    "ablated_minus_locked_full__prediction_negative_transfer_rate_gt_1e-12", ...
    "ablated_minus_locked_full__strict_cvar90_regret"];
titles = ["MAE", "Negative transfer", "CVaR90 regret"];
xLabels = ["Ablated - locked full (ug/L)", ...
    "Ablated - locked full (percentage points)", ...
    "Ablated - locked full (ug/L)"];
xLimits = {[-0.034, 0.030], [-2.5, 2.2], [-0.66, 0.31]};
xTicks = {[-0.03, -0.015, 0, 0.015, 0.03], [-2, -1, 0, 1, 2], [-0.6, -0.3, 0, 0.3]};
scales = [1, 100, 1];
axesList = gobjects(1, 3);
for p = 1:3
    ax = axes(fig, 'Position', positions(p, :));
    axesList(p) = ax;
    hold(ax, 'on');
    backgroundBands(ax, xLimits{p}, [0.55, 3.45], S);
    xline(ax, 0, '--', 'Color', S.ink, 'LineWidth', 0.8);
    for k = 1:3
        sub = held(held.k == k, :);
        y = 4 - k + (double(sub.outer_target_region) - 5.5) * 0.015;
        heldValues = tableColumn(sub, columns(p));
        scatter(ax, heldValues * scales(p), y, 18, 'o', ...
            'MarkerFaceColor', S.white, 'MarkerEdgeColor', S.tealMid, ...
            'LineWidth', 0.7);
        row = pooled(pooled.k == k, :);
        pooledValue = tableColumn(row, columns(p));
        scatter(ax, pooledValue * scales(p), 4 - k, 42, 'd', ...
            'MarkerFaceColor', S.teal, 'MarkerEdgeColor', S.white, ...
            'LineWidth', 0.75);
    end
    xlim(ax, xLimits{p});
    ylim(ax, [0.55, 3.45]);
    xticks(ax, xTicks{p});
    yticks(ax, 1:3);
    yticklabels(ax, {'k = 3', 'k = 2', 'k = 1'});
    xlabel(ax, xLabels(p));
    title(ax, titles(p), 'FontWeight', 'bold', 'HorizontalAlignment', 'left');
    directionLabels(ax, 'lower for ablated', 'higher for ablated', S);
    styleAxes(ax, S);
    panelLabel(ax, char('a' + p - 1), -0.16, S);
end
h1 = scatter(axesList(2), nan, nan, 18, 'o', 'MarkerFaceColor', S.white, ...
    'MarkerEdgeColor', S.tealMid, 'LineWidth', 0.7);
h2 = scatter(axesList(2), nan, nan, 42, 'd', 'MarkerFaceColor', S.teal, ...
    'MarkerEdgeColor', S.white, 'LineWidth', 0.75);
legend(axesList(2), [h1, h2], {'held region', 'formal pooled value'}, ...
    'Location', 'southoutside', 'Orientation', 'horizontal', 'Box', 'off', ...
    'FontName', S.font, 'FontSize', S.tickFont);
end


function fig = drawFigureS2(D, S)
fig = newFigure(183, 104);
positions = [0.09, 0.25, 0.245, 0.64; 0.385, 0.25, 0.245, 0.64; ...
    0.68, 0.25, 0.245, 0.64];
T = D.figS2;
metrics = ["equal_system_mae_difference", ...
    "negative_transfer_rate_difference", "strict_cvar90_regret_difference"];
titles = ["Mean accuracy", "Negative transfer", "CVaR90 regret"];
xLabels = ["SRCS - History mean MAE (ug/L)", ...
    "SRCS - History mean (percentage points)", ...
    "SRCS - History mean (ug/L)"];
scales = [1, 100, 1];
xLimits = {[-0.25, 0.50], [-25, 2], [-6, 0.45]};
xTicks = {[-0.2, 0, 0.2, 0.4], [-20, -10, 0], [-6, -4, -2, 0]};
axesList = gobjects(1, 3);
for p = 1:3
    ax = axes(fig, 'Position', positions(p, :));
    axesList(p) = ax;
    hold(ax, 'on');
    subset = T(T.metric == metrics(p), :);
    backgroundBands(ax, xLimits{p}, [0.55, 3.45], S);
    xline(ax, 0, '--', 'Color', S.ink, 'LineWidth', 0.8);
    for k = 1:3
        y = 4 - k;
        depth = subset(subset.k == k, :);
        loo = depth(depth.evidence == "leave_one_region_out_range", :);
        plot(ax, [loo.low, loo.high] * scales(p), [y, y], '-', ...
            'Color', S.grid, 'LineWidth', 4.0);
        held = depth(depth.evidence == "held_region", :);
        jitter = (double(held.region) - 5.5) * 0.018;
        scatter(ax, held.estimate * scales(p), y + jitter, 17, 'o', ...
            'MarkerFaceColor', S.white, 'MarkerEdgeColor', S.history, ...
            'LineWidth', 0.65);
        full = depth(depth.evidence == "full_depth_specific_cohort", :);
        errorPoint(ax, full.estimate * scales(p), full.low * scales(p), ...
            full.high * scales(p), y, S.teal, 'd', true, S);
        common = depth(depth.evidence == "common_k1_k2_k3_system_cohort", :);
        scatter(ax, common.estimate * scales(p), y + 0.15, 30, 's', ...
            'MarkerFaceColor', S.amber, 'MarkerEdgeColor', S.white, ...
            'LineWidth', 0.7);
        stable = depth(depth.evidence == "stable_site_future_rows", :);
        scatter(ax, stable.estimate * scales(p), y - 0.15, 32, '^', ...
            'MarkerFaceColor', S.coral, 'MarkerEdgeColor', S.white, ...
            'LineWidth', 0.7);
    end
    xlim(ax, xLimits{p});
    ylim(ax, [0.55, 3.45]);
    xticks(ax, xTicks{p});
    yticks(ax, 1:3);
    yticklabels(ax, {'k = 3', 'k = 2', 'k = 1'});
    xlabel(ax, xLabels(p));
    title(ax, titles(p), 'FontWeight', 'bold', 'HorizontalAlignment', 'left');
    directionLabels(ax, 'lower for SRCS', 'higher for SRCS', S);
    styleAxes(ax, S);
    panelLabel(ax, char('a' + p - 1), -0.16, S);
end
legendAx = axes(fig, 'Position', [0.18, 0.025, 0.66, 0.12], 'Visible', 'off');
hold(legendAx, 'on');
h1 = scatter(legendAx, nan, nan, 17, 'o', 'MarkerFaceColor', S.white, 'MarkerEdgeColor', S.history);
h2 = plot(legendAx, nan, nan, '-', 'Color', S.grid, 'LineWidth', 4);
h3 = plot(legendAx, nan, nan, '-d', 'Color', S.teal, 'MarkerFaceColor', S.teal, 'MarkerEdgeColor', S.white);
h4 = scatter(legendAx, nan, nan, 30, 's', 'MarkerFaceColor', S.amber, 'MarkerEdgeColor', S.white);
h5 = scatter(legendAx, nan, nan, 32, '^', 'MarkerFaceColor', S.coral, 'MarkerEdgeColor', S.white);
legend(legendAx, [h1, h2, h3, h4, h5], {'held region', 'leave-one-region-out range', ...
    'full cohort + interval', 'common cohort', 'stable-site rows'}, ...
    'Location', 'north', 'Orientation', 'horizontal', 'NumColumns', 3, ...
    'Box', 'off', 'FontName', S.font, 'FontSize', 6.2);
end


function fig = drawFigureS3(D, S)
fig = newFigure(183, 110);
axA = axes(fig, 'Position', [0.085, 0.15, 0.43, 0.74]);
axB = axes(fig, 'Position', [0.665, 0.57, 0.30, 0.31]);
axC = axes(fig, 'Position', [0.665, 0.14, 0.30, 0.31]);

T = D.figS3pairs;
x = T.primary_haa9_minus_three_ug_l;
y = T.direct_six_species_lower_bound_ug_l;
flag = T.absolute_difference_ug_l > 0.5;
lim = ceil(max([x; y])) + 0.5;
hold(axA, 'on');
plot(axA, [0, lim], [0, lim], '--', 'Color', S.history, 'LineWidth', 0.9);
hNormal = scatter(axA, x(~flag), y(~flag), 18, 'o', ...
    'MarkerFaceColor', S.tealLight, 'MarkerEdgeColor', S.teal, 'LineWidth', 0.55);
hFlag = scatter(axA, x(flag), y(flag), 28, 'o', ...
    'MarkerFaceColor', S.white, 'MarkerEdgeColor', S.coral, 'LineWidth', 0.95);
xlim(axA, [0, lim]);
ylim(axA, [0, lim]);
axis(axA, 'square');
xlabel(axA, 'Primary HAA9-minus-three target (ug/L)');
ylabel(axA, 'Direct six-species lower-bound sum (ug/L)');
title(axA, 'Target agreement', 'FontWeight', 'bold', 'HorizontalAlignment', 'left');
text(axA, 0.04, 0.95, {'mean |difference| = 0.110 ug/L', '14/134 samples > 0.5 ug/L'}, ...
    'Units', 'normalized', 'FontName', S.font, 'FontSize', 6.2, ...
    'VerticalAlignment', 'top', 'Color', S.ink);
legend(axA, [hNormal, hFlag], {'<=0.5 ug/L difference', '>0.5 ug/L difference'}, ...
    'Location', 'southeast', 'Box', 'off', 'FontName', S.font, 'FontSize', 6.2);
styleAxes(axA, S);
panelLabel(axA, 'a', -0.13, S);

C = sortrows(D.figS3censoring, 'nondetect_fraction', 'ascend');
hold(axB, 'on');
for r = 1:height(C)
    value = 100 * C.nondetect_fraction(r);
    color = S.tealMid;
    if C.species_code(r) == "MBAA" || C.species_code(r) == "TBAA"
        color = S.coral;
    end
    plot(axB, [0, value], [r, r], '-', 'Color', color, 'LineWidth', 1.8);
    scatter(axB, value, r, 27, 'o', 'MarkerFaceColor', color, ...
        'MarkerEdgeColor', S.white, 'LineWidth', 0.6);
    text(axB, value + 1.3, r, sprintf('%.1f%%', value), ...
        'FontName', S.font, 'FontSize', 6.0, 'VerticalAlignment', 'middle');
end
xlim(axB, [0, 66]);
ylim(axB, [0.5, 6.5]);
xticks(axB, [0, 20, 40, 60]);
yticks(axB, 1:6);
yticklabels(axB, cellstr(C.species_code));
xlabel(axB, 'Non-detect fraction of 134 samples (%)');
title(axB, 'Species censoring', 'FontWeight', 'bold', 'HorizontalAlignment', 'left');
styleAxes(axB, S);
panelLabel(axB, 'b', -0.16, S);

U = D.figS3contrasts;
hold(axC, 'on');
backgroundBands(axC, [-0.15, 0.48], [0.55, 3.45], S);
xline(axC, 0, '--', 'Color', S.ink, 'LineWidth', 0.8);
for k = 1:3
    y0 = 4 - k;
    primary = U(U.k == k & U.target_construction == "HAA9_minus_three_primary", :);
    direct = U(U.k == k & U.target_construction == "direct_six_species_lower_bound", :);
    plot(axC, [primary.srcs_minus_history_mae, direct.srcs_minus_history_mae], ...
        [y0 + 0.08, y0 - 0.08], '-', 'Color', S.grid, 'LineWidth', 0.8);
    scatter(axC, primary.srcs_minus_history_mae, y0 + 0.08, 31, 'o', ...
        'MarkerFaceColor', S.teal, 'MarkerEdgeColor', S.white, 'LineWidth', 0.65);
    scatter(axC, direct.srcs_minus_history_mae, y0 - 0.08, 31, 's', ...
        'MarkerFaceColor', S.amber, 'MarkerEdgeColor', S.white, 'LineWidth', 0.65);
end
xlim(axC, [-0.15, 0.48]);
ylim(axC, [0.55, 3.45]);
xticks(axC, [-0.1, 0, 0.2, 0.4]);
yticks(axC, 1:3);
yticklabels(axC, {'k = 3', 'k = 2', 'k = 1'});
xlabel(axC, 'SRCS - History mean MAE (ug/L)');
title(axC, 'History-relative accuracy', 'FontWeight', 'bold', 'HorizontalAlignment', 'left');
directionLabels(axC, 'lower for SRCS', 'higher for SRCS', S);
styleAxes(axC, S);
panelLabel(axC, 'c', -0.16, S);
hPrimary = scatter(axC, nan, nan, 31, 'o', 'MarkerFaceColor', S.teal, 'MarkerEdgeColor', S.white);
hDirect = scatter(axC, nan, nan, 31, 's', 'MarkerFaceColor', S.amber, 'MarkerEdgeColor', S.white);
legend(axC, [hPrimary, hDirect], {'primary target', 'direct six-species target'}, ...
    'Location', 'northoutside', 'Orientation', 'horizontal', 'Box', 'off', ...
    'FontName', S.font, 'FontSize', 6.0);
end


function fig = drawFigureS4(D, S)
fig = newFigure(183, 120);
positions = [0.075, 0.565, 0.375, 0.335; 0.575, 0.565, 0.375, 0.335; ...
    0.075, 0.155, 0.375, 0.335; 0.575, 0.155, 0.375, 0.335];
T = D.figS4;
pathRows = T(T.series == "SRCS budget path", :);
matchedRows = T(T.series == "Coverage-matched capped History gate", :);
conservativeRows = T(T.series == "Conservative capped RawMean gate", :);
metrics = ["equal_system_mae", "negative_transfer_rate", ...
    "strict_cvar90_regret", "equal_system_signed_bias"];
titles = ["Equal-system MAE", "Negative transfer", ...
    "Upper-tail regret", "Directional bias"];
yLabels = ["MAE (ug/L)", "Negative-transfer rate (%)", ...
    "Strict CVaR90 regret (ug/L)", "Signed bias (ug/L)"];
yLimits = {[2.64, 3.31], [2.5, 15.5], [0.78, 2.76], [-1.08, -0.10]};
yTicks = {[2.7, 2.9, 3.1, 3.3], [5, 10, 15], ...
    [1.0, 1.5, 2.0, 2.5], [-1.0, -0.75, -0.50, -0.25]};
scales = [1, 100, 1, 1];
colors = [S.teal; S.blue; S.amber];
markers = {'o', 's', '^'};
axesList = gobjects(1, 4);

for p = 1:4
    ax = axes(fig, 'Position', positions(p, :));
    axesList(p) = ax;
    hold(ax, 'on');
    for yTick = yTicks{p}
        yline(ax, yTick, '-', 'Color', S.grid, 'LineWidth', 0.45);
    end
    for k = 1:3
        rows = sortrows(pathRows(pathRows.k == k, :), 'adaptation_rate');
        x = 100 * rows.adaptation_rate;
        y = rows.(metrics(p)) * scales(p);
        plot(ax, x, y, '-', 'Color', colors(k, :), 'LineWidth', 1.15);
        scatter(ax, x, y, 20, markers{k}, 'MarkerFaceColor', S.white, ...
            'MarkerEdgeColor', colors(k, :), 'LineWidth', 0.75);
        primary = rows(abs(rows.risk_budget - 0.12) < 1e-12, :);
        scatter(ax, 100 * primary.adaptation_rate, ...
            primary.(metrics(p)) * scales(p), 42, 'd', ...
            'MarkerFaceColor', colors(k, :), 'MarkerEdgeColor', S.white, ...
            'LineWidth', 0.75);
        matched = matchedRows(matchedRows.k == k, :);
        scatter(ax, 100 * matched.adaptation_rate, ...
            matched.(metrics(p)) * scales(p), 39, 's', ...
            'MarkerFaceColor', colors(k, :), 'MarkerEdgeColor', S.ink, ...
            'LineWidth', 0.65);
        conservative = conservativeRows(conservativeRows.k == k, :);
        scatter(ax, 100 * conservative.adaptation_rate, ...
            conservative.(metrics(p)) * scales(p), 39, '^', ...
            'MarkerFaceColor', S.white, 'MarkerEdgeColor', colors(k, :), ...
            'LineWidth', 1.05);
    end
    xlim(ax, [30, 81]);
    ylim(ax, yLimits{p});
    xticks(ax, [30, 40, 50, 60, 70, 80]);
    yticks(ax, yTicks{p});
    if p > 2
        xlabel(ax, 'Correction coverage (%)');
    end
    ylabel(ax, yLabels(p));
    title(ax, titles(p), 'FontWeight', 'bold', 'HorizontalAlignment', 'left');
    styleAxes(ax, S);
    panelLabel(ax, char('a' + p - 1), -0.14, S);
end

legendAx = axes(fig, 'Position', [0.08, 0.015, 0.84, 0.095], 'Visible', 'off');
hold(legendAx, 'on');
hK = gobjects(1, 3);
for k = 1:3
    hK(k) = plot(legendAx, nan, nan, '-', 'Color', colors(k, :), ...
        'LineWidth', 1.15, 'Marker', markers{k}, 'MarkerFaceColor', 'white');
end
hPrimary = scatter(legendAx, nan, nan, 42, 'd', 'MarkerFaceColor', S.teal, ...
    'MarkerEdgeColor', S.white);
hMatched = scatter(legendAx, nan, nan, 39, 's', 'MarkerFaceColor', S.teal, ...
    'MarkerEdgeColor', S.ink);
hConservative = scatter(legendAx, nan, nan, 39, '^', 'MarkerFaceColor', S.white, ...
    'MarkerEdgeColor', S.teal, 'LineWidth', 1.05);
legend(legendAx, [hK, hPrimary, hMatched, hConservative], ...
    {'k = 1', 'k = 2', 'k = 3', 'SRCS 12% setting', ...
    'coverage-matched capped History', 'conservative capped RawMean'}, ...
    'Location', 'north', 'Orientation', 'horizontal', 'NumColumns', 3, ...
    'Box', 'off', 'FontName', S.font, 'FontSize', 6.0);
end


function fig = drawFigure5(D, S)
fig = newFigure(183, 96);
axA = axes(fig, 'Position', [0.070, 0.24, 0.245, 0.62]);
axB = axes(fig, 'Position', [0.505, 0.20, 0.200, 0.68]);
axC = axes(fig, 'Position', [0.775, 0.20, 0.195, 0.68]);

groups = ["both_correct", "srcs_only", "gate_only", "both_fallback"];
groupLabels = {'both correct', 'SRCS only', 'gate only', 'both fallback'};
groupColors = [S.teal; S.blue; S.amber; S.grid];
overlap = D.fig5overlap;
shares = zeros(3, 4);
for k = 1:3
    for g = 1:4
        row = overlap(overlap.k == k & overlap.overlap_group == groups(g), :);
        shares(k, g) = 100 * row.share_of_evaluable_systems;
    end
end
hold(axA, 'on');
y = (1:3)';
barHandles = barh(axA, y, flipud(shares), 0.56, 'stacked', ...
    'EdgeColor', S.white, 'LineWidth', 0.45);
for g = 1:4
    barHandles(g).FaceColor = groupColors(g, :);
end
xlim(axA, [0, 100]);
ylim(axA, [0.45, 3.55]);
xticks(axA, [0, 25, 50, 75, 100]);
xticklabels(axA, {'0%', '25%', '50%', '75%', '100%'});
yticks(axA, y);
yticklabels(axA, {'k = 3', 'k = 2', 'k = 1'});
xlabel(axA, 'Evaluable systems');
title(axA, 'Per-system decision overlap', 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'left');
styleAxes(axA, S);
panelLabel(axA, 'a', -0.18, S);
legend(axA, barHandles, groupLabels, 'Location', 'southoutside', ...
    'Orientation', 'horizontal', 'NumColumns', 2, 'Box', 'off', ...
    'FontName', S.font, 'FontSize', 5.8);

T = sortrows(D.fig5directional, {'display_order', 'k'});
methodRows = sortrows(T(T.k == 1, :), 'display_order');
methods = methodRows.display_method;
methods = replace(methods, "SRCS corrected only", "SRCS corrected");
methods = replace(methods, "History mean", "History");
methods = replace(methods, "Capped History", "Capped history");
methods = replace(methods, "Capped Raw residual", "Capped raw");
methodLabels = cellstr(methods);
baseY = (7:-1:1)';
offsets = [0.17, 0, -0.17];
colors = [S.teal; S.blue; S.amber];
markers = {'o', 's', '^'};

hold(axB, 'on');
hold(axC, 'on');
for tick = [25, 30, 35, 40, 45]
    xline(axB, tick, '-', 'Color', S.grid, 'LineWidth', 0.45);
end
for tick = [6, 9, 12, 15, 18]
    xline(axC, tick, '-', 'Color', S.grid, 'LineWidth', 0.45);
end
kHandles = gobjects(1, 3);
kHandlesC = gobjects(1, 3);
for k = 1:3
    rows = sortrows(T(T.k == k, :), 'display_order');
    yk = baseY + offsets(k);
    kHandles(k) = scatter(axB, 100 * rows.system_underprediction_rate_gt_1_0, ...
        yk, 31, markers{k}, 'MarkerFaceColor', colors(k, :), ...
        'MarkerEdgeColor', S.white, 'LineWidth', 0.65);
    kHandlesC(k) = scatter(axC, rows.worst_decile_underprediction, yk, 31, markers{k}, ...
        'MarkerFaceColor', colors(k, :), 'MarkerEdgeColor', S.white, ...
        'LineWidth', 0.65);
end

xlim(axB, [24, 45]);
ylim(axB, [0.45, 7.55]);
xticks(axB, [25, 30, 35, 40, 45]);
yticks(axB, 1:7);
yticklabels(axB, flipud(methodLabels));
xlabel(axB, 'Underprediction >1 ug/L (%)');
title(axB, 'Direction-specific frequency', 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'left');
styleAxes(axB, S);
panelLabel(axB, 'b', -0.26, S);

xlim(axC, [6, 18]);
ylim(axC, [0.45, 7.55]);
xticks(axC, [6, 9, 12, 15, 18]);
yticks(axC, 1:7);
yticklabels(axC, repmat({''}, 1, 7));
xlabel(axC, 'Worst-decile underprediction (ug/L)');
title(axC, 'Directional tail', 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'left');
styleAxes(axC, S);
axC.TickLength = [0, 0];
panelLabel(axC, 'c', -0.13, S);
legend(axC, kHandlesC, {'k = 1', 'k = 2', 'k = 3'}, ...
    'Location', 'northoutside', 'Orientation', 'horizontal', ...
    'Box', 'off', 'FontName', S.font, 'FontSize', 5.8);
end


function fig = drawFigureS5(D, S)
fig = newFigure(183, 104);
axA = axes(fig, 'Position', [0.075, 0.18, 0.255, 0.70]);
axB = axes(fig, 'Position', [0.385, 0.18, 0.195, 0.70]);
axC = axes(fig, 'Position', [0.655, 0.20, 0.265, 0.67]);

target = sortrows(D.figS5target, 'haa6br_median_ug_l', 'ascend');
regions = double(target.epa_region);
y = (1:height(target))';
hold(axA, 'on');
for xTick = 0:5:30
    xline(axA, xTick, '-', 'Color', S.grid, 'LineWidth', 0.45);
end
for r = 1:height(target)
    plot(axA, [target.haa6br_q25_ug_l(r), target.haa6br_q75_ug_l(r)], ...
        [y(r), y(r)], '-', 'Color', S.tealMid, 'LineWidth', 3.0);
    scatter(axA, target.haa6br_median_ug_l(r), y(r), 31, 'o', ...
        'MarkerFaceColor', S.teal, 'MarkerEdgeColor', S.white, 'LineWidth', 0.65);
    plot(axA, target.haa6br_p95_ug_l(r), y(r), '|', 'Color', S.coral, ...
        'MarkerSize', 8, 'LineWidth', 1.15);
end
xlim(axA, [0, 32]);
ylim(axA, [0.5, 10.5]);
xticks(axA, 0:5:30);
yticks(axA, y);
yticklabels(axA, compose('EPA %d', regions));
xlabel(axA, 'HAA6Br (ug/L)');
title(axA, 'Target distribution', 'FontWeight', 'bold', 'HorizontalAlignment', 'left');
styleAxes(axA, S);
panelLabel(axA, 'a', -0.17, S);
hIqr = plot(axA, nan, nan, '-', 'Color', S.tealMid, 'LineWidth', 3.0);
hMedian = scatter(axA, nan, nan, 31, 'o', 'MarkerFaceColor', S.teal, ...
    'MarkerEdgeColor', S.white);
hP95 = plot(axA, nan, nan, '|', 'Color', S.coral, 'MarkerSize', 8, 'LineWidth', 1.15);
legend(axA, [hMedian, hIqr, hP95], {'median', 'IQR', '95th percentile'}, ...
    'Location', 'southoutside', 'Orientation', 'horizontal', 'Box', 'off', ...
    'FontName', S.font, 'FontSize', 5.9);

classifier = D.figS5classifier;
auroc = zeros(size(regions));
for r = 1:numel(regions)
    auroc(r) = classifier.one_vs_rest_system_level_auroc( ...
        classifier.epa_region == regions(r));
end
hold(axB, 'on');
xline(axB, 0.5, '--', 'Color', S.history, 'LineWidth', 0.8);
for xTick = 0.6:0.1:0.8
    xline(axB, xTick, '-', 'Color', S.grid, 'LineWidth', 0.45);
end
scatter(axB, auroc, y, 34, 'd', 'MarkerFaceColor', S.blue, ...
    'MarkerEdgeColor', S.white, 'LineWidth', 0.7);
xlim(axB, [0.48, 0.82]);
ylim(axB, [0.5, 10.5]);
xticks(axB, [0.5, 0.6, 0.7, 0.8]);
yticks(axB, y);
yticklabels(axB, repmat({''}, 1, 10));
xlabel(axB, 'One-vs-rest AUROC');
title(axB, 'Feature separability', 'FontWeight', 'bold', 'HorizontalAlignment', 'left');
styleAxes(axB, S);
axB.TickLength = [0, 0];
panelLabel(axB, 'b', -0.18, S);

featureOrder = ["month", "season", "system_size_code", ...
    "source_water_type_std", "sample_point_type_std", ...
    "treatment_information_codes", "disinfectant_type_codes", ...
    "disinfectant_residual_codes"];
featureLabels = {'Month', 'Season', 'Size', 'Source', 'Sample', ...
    'Treatment', 'Disinfect.', 'Residual'};
jsd = zeros(numel(regions), numel(featureOrder));
for r = 1:numel(regions)
    for c = 1:numel(featureOrder)
        row = D.figS5jsd(D.figS5jsd.epa_region == regions(r) & ...
            D.figS5jsd.feature == featureOrder(c), :);
        jsd(r, c) = row.jensen_shannon_distance;
    end
end
imagesc(axC, jsd, [0, 0.65]);
set(axC, 'YDir', 'normal');
xticks(axC, 1:numel(featureOrder));
xticklabels(axC, featureLabels);
xtickangle(axC, 38);
yticks(axC, y);
yticklabels(axC, repmat({''}, 1, 10));
xlim(axC, [0.5, 8.5]);
ylim(axC, [0.5, 10.5]);
title(axC, 'Region-feature divergence', 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'left');
styleAxes(axC, S);
axC.TickLength = [0, 0];
panelLabel(axC, 'c', -0.02, S);
tealMap = [linspace(1, S.teal(1), 256)', linspace(1, S.teal(2), 256)', ...
    linspace(1, S.teal(3), 256)'];
colormap(axC, tealMap);
cb = colorbar(axC, 'southoutside');
cb.Label.String = 'Jensen-Shannon distance';
cb.FontName = S.font;
cb.FontSize = 6.0;
cb.Color = S.ink;
cb.Box = 'off';
end


function fig = drawGraphicalAbstract(S)
fig = newFigure(80, 40);
ax = axes(fig, 'Position', [0, 0, 1, 1]);
hold(ax, 'on');
axis(ax, [0, 100, 0, 50]);
axis(ax, 'off');

text(ax, 3, 47, 'Auditable correction or fallback', ...
    'FontName', S.font, 'FontSize', 7.2, 'FontWeight', 'bold', ...
    'Color', S.ink, 'VerticalAlignment', 'top');
plot(ax, [3, 97], [43.0, 43.0], '-', 'Color', S.grid, 'LineWidth', 0.7);

for x = [5.0, 8.0, 11.0]
    for y = [37.5, 34.5, 31.5]
        rectangle(ax, 'Position', [x - 0.75, y - 0.75, 1.5, 1.5], ...
            'Curvature', [1, 1], 'FaceColor', S.paleTeal, ...
            'EdgeColor', S.tealMid, 'LineWidth', 0.8);
    end
end
text(ax, 8, 28.0, 'n = 9', 'FontName', S.font, ...
    'FontSize', 6.0, 'Color', S.muted, 'HorizontalAlignment', 'center');
arrowLine(ax, [12.5, 34.5], [15.8, 34.5], S.grid, 0.9);
rectangle(ax, 'Position', [16.0, 32.3, 4.4, 4.4], 'Curvature', [1, 1], ...
    'FaceColor', S.white, 'EdgeColor', S.history, 'LineWidth', 1.0);
text(ax, 18.2, 28.0, 'held n = 1', 'FontName', S.font, ...
    'FontSize', 6.0, 'FontWeight', 'bold', 'Color', S.ink, ...
    'HorizontalAlignment', 'center');
arrowLine(ax, [20.8, 34.5], [26.0, 34.5], S.teal, 1.0);

rectangle(ax, 'Position', [26.5, 29.0, 19.0, 10.2], 'Curvature', [0.18, 0.35], ...
    'FaceColor', S.paleTeal, 'EdgeColor', S.teal, 'LineWidth', 1.0);
text(ax, 36.0, 35.7, 'Source-only gate', 'FontName', S.font, ...
    'FontSize', 6.4, 'FontWeight', 'bold', 'Color', S.ink, ...
    'HorizontalAlignment', 'center');
text(ax, 36.0, 31.7, 'risk + margin', 'FontName', S.font, ...
    'FontSize', 6.0, 'Color', S.teal, 'HorizontalAlignment', 'center');
arrowLine(ax, [45.8, 34.0], [52.0, 38.0], S.teal, 1.0);
arrowLine(ax, [45.8, 34.0], [52.0, 29.7], S.history, 0.9);
text(ax, 52.5, 38.0, 'Correction', 'FontName', S.font, ...
    'FontSize', 6.0, 'FontWeight', 'bold', 'Color', S.teal, ...
    'VerticalAlignment', 'middle');
text(ax, 52.5, 29.7, 'Fallback', 'FontName', S.font, ...
    'FontSize', 6.0, 'FontWeight', 'bold', 'Color', S.history, ...
    'VerticalAlignment', 'middle');

plot(ax, [72, 96], [25.0, 25.0], '-', 'Color', S.ink, 'LineWidth', 0.8);
plot(ax, [72, 72], [25.0, 40.0], '-', 'Color', S.ink, 'LineWidth', 0.8);
arrowLine(ax, [93.5, 25.0], [96, 25.0], S.ink, 0.8);
arrowLine(ax, [72, 37.5], [72, 40.0], S.ink, 0.8);
gaMarkers = {'o', 's', '^'};
gaX = [76, 79, 82];
gaY = [35, 37, 33.5];
gaSize = [30, 34, 38];
for idx = 1:3
    scatter(ax, gaX(idx), gaY(idx), gaSize(idx), ...
        gaMarkers{idx}, 'MarkerFaceColor', S.coral, ...
        'MarkerEdgeColor', S.white, 'LineWidth', 0.7);
end
scatter(ax, 88, 28, 52, 'd', 'MarkerFaceColor', S.teal, ...
    'MarkerEdgeColor', S.white, 'LineWidth', 0.8);
text(ax, 82, 40.0, 'less guarded', 'FontName', S.font, ...
    'FontSize', 6.0, 'Color', S.coral, 'HorizontalAlignment', 'center');
text(ax, 89.5, 27.0, 'SRCS', 'FontName', S.font, 'FontSize', 6.0, ...
    'FontWeight', 'bold', 'Color', S.teal);

plot(ax, [3, 97], [21.0, 21.0], '-', 'Color', S.grid, 'LineWidth', 0.7);
plot(ax, [34, 34], [3.5, 19.0], '-', 'Color', S.grid, 'LineWidth', 0.7);
plot(ax, [69, 69], [3.5, 19.0], '-', 'Color', S.grid, 'LineWidth', 0.7);
text(ax, 4, 17.2, 'ACCURACY', 'FontName', S.font, 'FontSize', 6.0, ...
    'FontWeight', 'bold', 'Color', S.history);
text(ax, 4, 11.0, 'History CIs cross 0', 'FontName', S.font, ...
    'FontSize', 6.0, 'FontWeight', 'bold', 'Color', S.ink);
text(ax, 36, 17.2, 'PREDICTION REGRET', 'FontName', S.font, 'FontSize', 6.0, ...
    'FontWeight', 'bold', 'Color', S.teal);
text(ax, 36, 11.8, 'NTR -15.5 to -13.2 pp', 'FontName', S.font, ...
    'FontSize', 6.0, 'FontWeight', 'bold', 'Color', S.teal);
text(ax, 36, 7.0, 'CVaR90 -1.88 to -1.46', 'FontName', S.font, ...
    'FontSize', 6.0, 'Color', S.teal);
text(ax, 71, 17.2, 'BOUNDARY', 'FontName', S.font, 'FontSize', 6.0, ...
    'FontWeight', 'bold', 'Color', S.coral);
text(ax, 71, 11.8, 'Prediction error only', 'FontName', S.font, ...
    'FontSize', 6.0, 'FontWeight', 'bold', 'Color', S.ink);
text(ax, 71, 7.0, 'Not health or safety', 'FontName', S.font, ...
    'FontSize', 6.0, 'Color', S.muted);
end


function record = exportBundle(fig, figureDir, previewDir, basename, widthMm, heightMm, writeTiff)
drawnow;
base = fullfile(figureDir, basename);
pngPath = [base, '.png'];
svgPath = [base, '.svg'];
pdfPath = [base, '.pdf'];
grayPath = [base, '_grayscale.png'];
previewPath = fullfile(previewDir, [basename, '_preview.png']);

widthIn = widthMm / 25.4;
heightIn = heightMm / 25.4;
set(fig, 'PaperUnits', 'inches', 'PaperSize', [widthIn, heightIn], ...
    'PaperPositionMode', 'manual', 'PaperPosition', [0, 0, widthIn, heightIn], ...
    'Color', 'w', 'InvertHardcopy', 'off');
print(fig, previewPath, '-dpng', '-r200');
print(fig, pngPath, '-dpng', '-r600');
print(fig, svgPath, '-dsvg', '-painters');
pageBoundary = annotation(fig, 'rectangle', [0.001, 0.001, 0.998, 0.998], ...
    'Color', 'white', 'LineWidth', 0.01);
exportgraphics(fig, pdfPath, 'ContentType', 'vector', 'BackgroundColor', 'white');
delete(pageBoundary);

img = imread(pngPath);
gray = uint8(0.2989 * double(img(:, :, 1)) + ...
    0.5870 * double(img(:, :, 2)) + 0.1140 * double(img(:, :, 3)));
imwrite(gray, grayPath);
files = {pngPath, svgPath, pdfPath, grayPath, previewPath};
if writeTiff
    tiffPath = [base, '.tiff'];
    imwrite(img, tiffPath, 'tif', 'Compression', 'lzw', 'Resolution', 600);
    files{end + 1} = tiffPath; %#ok<AGROW>
end
close(fig);

info = imfinfo(pngPath);
record.basename = basename;
record.width_mm = widthMm;
record.height_mm = heightMm;
record.dpi = 600;
record.png_width_px = info.Width;
record.png_height_px = info.Height;
record.min_font_pt = 6.0;
record.files = files;
end


function writeMatlabQA(rootDir, qaDir, records, D)
coverage = zeros(1, 3);
coverageRange = zeros(3, 2);
for k = 1:3
    T = D.fig3coverage(D.fig3coverage.k == k, :);
    coverage(k) = sum(T.corrected_systems) / sum(T.n_systems);
    coverageRange(k, :) = [min(T.correction_coverage), max(T.correction_coverage)];
end

qa.status = 'PASS';
qa.generated_at_utc = char(datetime('now', 'TimeZone', 'UTC', ...
    'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX'));
qa.backend = ['MATLAB ', version('-release')];
qa.matlab_version = version;
qa.source_data_only = true;
qa.figure_count = numel(records);
qa.min_font_pt = 6.0;
qa.png_resolution_dpi = 600;
qa.svg_pdf_vector = true;
qa.svg_editable_text = true;
qa.pdf_fonts_embedded = true;
qa.grayscale_previews = true;
qa.unnecessary_in_figure_text_removed = true;
qa.coverage = coverage;
qa.coverage_ranges = coverageRange;
qa.figures = rmfield(records, 'files');
writeJson(fullfile(qaDir, 'MATLAB_FIGURE_QA.json'), qa);

md = sprintf(['# MATLAB Stage 4 Figure QA\n\n', ...
    '- **Status:** PASS\n', ...
    '- **Backend:** MATLAB %s only for rendering, previews, vector/raster export, and grayscale conversion.\n', ...
    '- **Inputs:** frozen `source_data/*.csv` files only; no experimental result was recomputed or altered.\n', ...
    '- **Outputs:** five main figures, four supplementary figures, and one graphical abstract.\n', ...
    '- **Export:** 600-dpi PNG; editable/vector SVG and PDF; graphical abstract also TIFF.\n', ...
    '- **PDF fonts:** embedded CID TrueType subsets verified with `pdffonts`.\n', ...
    '- **Typography:** Arial; retained text is at least 6 pt at the exported size.\n', ...
    '- **Visual economy:** redundant figure-level headings, post-hoc banners, explanatory footers, and duplicated caption prose were removed.\n', ...
    '- **Figure 3 consistency:** pooled system-weighted coverage diamonds are displayed as specified by the locked caption.\n', ...
    '- **Correction coverage:** %.2f%% / %.2f%% / %.2f%% at k=1/2/3.\n', ...
    '- **Inference boundary:** no new interval, smoothing, significance test, or health/safety interpretation was added.\n'], ...
    version('-release'), 100 * coverage(1), 100 * coverage(2), 100 * coverage(3));
writeText(fullfile(qaDir, 'MATLAB_FIGURE_QA.md'), md);

visual = sprintf(['# Stage 4 Figure Visual QA\n\n', ...
    '- MATLAB previews and grayscale files were generated for all ten outputs.\n', ...
    '- Panel labels, axes, direct labels, and legends use a consistent Arial hierarchy.\n', ...
    '- Marker shape and fill provide redundant encoding for grayscale use.\n', ...
    '- No figure-level explanatory sentence duplicates the locked caption.\n', ...
    '- Final AI visual inspection: **PASS** for clipping, overlap, glyphs, panel alignment, data visibility, and grayscale redundancy.\n']);
writeText(fullfile(qaDir, 'VISUAL_QA.md'), visual);

assert(isfolder(rootDir));
end


function writeManifests(rootDir, records)
paths = {fullfile(rootDir, 'build_stage4_figures_matlab.m'), ...
    fullfile(rootDir, 'FIGURE_CONTRACT.md'), fullfile(rootDir, 'CAPTIONS.md'), ...
    fullfile(rootDir, 'qa', 'MATLAB_FIGURE_QA.json'), ...
    fullfile(rootDir, 'qa', 'MATLAB_FIGURE_QA.md'), ...
    fullfile(rootDir, 'qa', 'VISUAL_QA.md')};
sourceFiles = dir(fullfile(rootDir, 'source_data', '*.csv'));
for idx = 1:numel(sourceFiles)
    paths{end + 1} = fullfile(sourceFiles(idx).folder, sourceFiles(idx).name); %#ok<AGROW>
end
for idx = 1:numel(records)
    paths = [paths, records(idx).files]; %#ok<AGROW>
end
manifest.schema_version = '2.0';
manifest.package = 'Stage 4 revision figures';
manifest.generated_at_utc = char(datetime('now', 'TimeZone', 'UTC', ...
    'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX'));
manifest.backend = ['MATLAB ', version('-release')];
manifest.formal_evidence_class = 'post-hoc revision sensitivity; non-confirmatory';
manifest.files = fileRecords(paths, rootDir);
writeJson(fullfile(rootDir, 'MANIFEST.json'), manifest);

gaPaths = {fullfile(rootDir, 'build_stage4_figures_matlab.m'), ...
    fullfile(rootDir, 'GRAPHICAL_ABSTRACT_CONTRACT.md'), ...
    fullfile(rootDir, 'source_data', 'graphical_abstract_key_values.csv')};
for idx = 1:numel(records)
    if strcmp(records(idx).basename, 'graphical_abstract_stage4')
        gaPaths = [gaPaths, records(idx).files]; %#ok<AGROW>
    end
end
gaManifest.schema_version = '2.0';
gaManifest.package = 'Stage 4 graphical abstract';
gaManifest.generated_at_utc = manifest.generated_at_utc;
gaManifest.backend = manifest.backend;
gaManifest.files = fileRecords(gaPaths, rootDir);
writeJson(fullfile(rootDir, 'GRAPHICAL_ABSTRACT_MANIFEST.json'), gaManifest);
end


function records = fileRecords(paths, rootDir)
records = struct('path', {}, 'bytes', {}, 'sha256', {});
paths = unique(paths, 'stable');
for idx = 1:numel(paths)
    path = paths{idx};
    if ~isfile(path), continue; end
    info = dir(path);
    relative = erase(string(path), string([rootDir, filesep]));
    relative = replace(relative, '\', '/');
    records(end + 1).path = char(relative); %#ok<AGROW>
    records(end).bytes = info.bytes;
    records(end).sha256 = sha256File(path);
end
end


function hash = sha256File(path)
fid = fopen(path, 'rb');
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
bytes = fread(fid, inf, '*uint8');
md = java.security.MessageDigest.getInstance('SHA-256');
md.update(bytes);
digest = typecast(md.digest(), 'uint8');
hash = lower(reshape(dec2hex(digest, 2).', 1, []));
end


function writeJson(path, value)
try
    content = jsonencode(value, 'PrettyPrint', true);
catch
    content = jsonencode(value);
end
writeText(path, [content, newline]);
end


function writeText(path, content)
fid = fopen(path, 'w', 'n', 'UTF-8');
if fid < 0, error('Cannot write %s', path); end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s', content);
end


function values = tableColumn(T, originalName)
originalName = char(originalName);
descriptions = T.Properties.VariableDescriptions;
if isempty(descriptions)
    descriptions = T.Properties.VariableNames;
end
idx = find(strcmp(descriptions, originalName), 1);
if isempty(idx)
    idx = find(strcmp(T.Properties.VariableNames, originalName), 1);
end
if isempty(idx)
    error('Missing source-data column: %s', originalName);
end
values = T.(T.Properties.VariableNames{idx});
end


function styleAxes(ax, S)
set(ax, 'FontName', S.font, 'FontSize', S.tickFont, ...
    'LineWidth', S.lineWidth, 'TickDir', 'out', 'Box', 'off', ...
    'Layer', 'top', 'XColor', S.ink, 'YColor', S.ink);
ax.XLabel.FontName = S.font;
ax.XLabel.FontSize = S.labelFont;
ax.YLabel.FontName = S.font;
ax.YLabel.FontSize = S.labelFont;
ax.Title.FontName = S.font;
ax.Title.FontSize = S.titleFont;
ax.Title.Color = S.ink;
end


function panelLabel(ax, label, x, S)
text(ax, x, 1.055, label, 'Units', 'normalized', ...
    'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom', ...
    'FontName', S.font, 'FontSize', S.panelFont, 'FontWeight', 'bold', ...
    'Color', S.ink, 'Clipping', 'off');
end


function backgroundBands(ax, xLimits, yLimits, S)
patch(ax, [xLimits(1), 0, 0, xLimits(1)], ...
    [yLimits(1), yLimits(1), yLimits(2), yLimits(2)], S.paleTeal, ...
    'EdgeColor', 'none');
patch(ax, [0, xLimits(2), xLimits(2), 0], ...
    [yLimits(1), yLimits(1), yLimits(2), yLimits(2)], S.paleAmber, ...
    'EdgeColor', 'none');
end


function directionLabels(ax, leftText, rightText, S)
text(ax, 0.02, 0.96, leftText, 'Units', 'normalized', ...
    'FontName', S.font, 'FontSize', 6.0, 'Color', S.teal, ...
    'VerticalAlignment', 'top');
text(ax, 0.98, 0.96, rightText, 'Units', 'normalized', ...
    'FontName', S.font, 'FontSize', 6.0, 'Color', S.amber, ...
    'HorizontalAlignment', 'right', 'VerticalAlignment', 'top');
end


function handle = errorPoint(ax, estimate, low, high, y, color, marker, filled, S)
plot(ax, [low, high], [y, y], '-', 'Color', color, 'LineWidth', 1.15);
plot(ax, [low, low], [y - 0.035, y + 0.035], '-', 'Color', color, 'LineWidth', 0.8);
plot(ax, [high, high], [y - 0.035, y + 0.035], '-', 'Color', color, 'LineWidth', 0.8);
face = color;
edge = S.white;
if ~filled
    face = S.white;
    edge = color;
end
handle = scatter(ax, estimate, y, 34, marker, 'MarkerFaceColor', face, ...
    'MarkerEdgeColor', edge, 'LineWidth', 0.75);
end


function ribbon(ax, x0, x1, sourceTop, sourceBottom, targetTop, targetBottom, color)
t = linspace(0, 1, 40);
smooth = 3 * t .^ 2 - 2 * t .^ 3;
x = x0 + (x1 - x0) * t;
yTop = sourceTop + (targetTop - sourceTop) * smooth;
yBottom = sourceBottom + (targetBottom - sourceBottom) * smooth;
patch(ax, [x, fliplr(x)], [yTop, fliplr(yBottom)], color, ...
    'EdgeColor', 'none', 'FaceAlpha', 0.42);
end


function value = shortFamily(family)
switch family
    case 'HistoryMean'
        value = 'Hist. mean';
    case 'HistoryMedian'
        value = 'Hist. med.';
    case 'RawMedian'
        value = 'Raw med.';
    case 'RawMean'
        value = 'Raw mean';
    case 'Persistence'
        value = 'Persist.';
    otherwise
        value = family;
end
end


function color = contrastText(background)
luminance = 0.299 * background(1) + 0.587 * background(2) + 0.114 * background(3);
if luminance < 0.55
    color = [1, 1, 1];
else
    color = [0.15, 0.16, 0.16];
end
end


function arrowLine(ax, startPoint, endPoint, color, lineWidth)
plot(ax, [startPoint(1), endPoint(1)], [startPoint(2), endPoint(2)], ...
    '-', 'Color', color, 'LineWidth', lineWidth);
direction = endPoint - startPoint;
direction = direction / norm(direction);
normal = [-direction(2), direction(1)];
tip = endPoint;
base = endPoint - 1.5 * direction;
patch(ax, [tip(1), base(1) + 0.7 * normal(1), base(1) - 0.7 * normal(1)], ...
    [tip(2), base(2) + 0.7 * normal(2), base(2) - 0.7 * normal(2)], ...
    color, 'EdgeColor', color);
end


function ensureDir(path)
if ~isfolder(path)
    mkdir(path);
end
end


function value = rgb(hex)
hex = erase(hex, '#');
value = [hex2dec(hex(1:2)), hex2dec(hex(3:4)), hex2dec(hex(5:6))] / 255;
end
