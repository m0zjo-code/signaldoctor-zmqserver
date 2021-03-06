%% Load Data
T = readtable('PeakDetectResults.csv');
A = table2array(T(:,2:4));

%% Plot data

figure(1)
hBar = bar(log(1000000*A(:,2)));
xt = get(gca, 'XTick');
set(gca, 'XTick', xt, 'XTickLabel', {
    'scipy-cwt'
    'scipy-argrelextrema'
    'scipy-findpeaks'
    'detect-peaks-md'
    'peakutils'
    'peakdetect-sb'
    'findpeaks-js'
    'detect-peaks-tb'
})
xtickangle(45)

grid on
grid minor

ylabel('log(Calculation Time(us))')
xlabel('Peak Detection Algorithms','FontSize',12)
title('Peak Detection Computation Time Results')