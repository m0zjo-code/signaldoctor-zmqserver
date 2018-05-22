% From https://stackoverflow.com/questions/28831781/qpsk-constellation-diagram

num_symbols = 1e4;
int_symbols = randi([1, 4], 1, num_symbols);
A = 1;
qpsk_symbols = zeros(size(int_symbols));
qpsk_symbols(int_symbols == 1) =   A + 1i*A;
qpsk_symbols(int_symbols == 2) =   A - 1i*A;
qpsk_symbols(int_symbols == 3) = - A + 1i*A;
qpsk_symbols(int_symbols == 4) = - A - 1i*A;
tx_sig = qpsk_symbols;

snr = 20; %// in dB
rx_sig = awgn(tx_sig, snr, 'measured');

rx_sig = rx_sig + 0.5 + 0.25i;

ptsx = [1 1 -1 -1];
ptsy = [-1 1 1 -1];


fh2 = figure;
plot_lims = [-2 2];
hold on
plot(real(rx_sig), imag(rx_sig), '.');
plot(ptsx, ptsy, 'rx')
grid on
grid minor
xlim(plot_lims);
ylim(plot_lims);
title(['QPSK constellation at an SNR of ' num2str(snr) ' dB and DC offset of 0.5+0.25j']);
xlabel('real part');
ylabel('imaginary part');