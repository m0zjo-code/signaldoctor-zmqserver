load('searchbuffer.mat')

log_buffer = 10*log(buffer_abs)+0.00001;
findpeaks(log_buffer, 'MinPeakProminence',10)
xlabel('Search Vector/Samples')
ylabel('10log(Amplitude/WHz^{-1})')
title('findpeaks() Example Usage')
xlim([3000 4000])
ylim([-70 35])