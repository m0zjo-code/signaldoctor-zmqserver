%% Load Data
A = dlmread ('PeakDetectResults.csv',',')
%% Plot data

figure(1)
hBar = bar(sqrt(A(:,3)));
xt = get(gca, 'XTick');
set(gca, 'XTick', xt, 'XTickLabel', {'scipy_cwt'
'scipy_argrelextrema'
'scipy_findpeaks'
'detect_peaks_tes'
'peakutils_test'
'peakdetect_test'
'findpeaks_test'
'tb_detect_peaks_test'})
xtickangle(45)

ylabel('sqrt(Time(s))')
xlabel('Peak Detection Algorithms')
title('Peak Detection Computation Times')