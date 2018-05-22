qpsk = comm.QPSKModulator;
sps = 16;
txfilter = comm.RaisedCosineTransmitFilter('Shape','Normal', ...
    'RolloffFactor',0.22, ...
    'FilterSpanInSymbols',20, ...
    'OutputSamplesPerSymbol',sps);

txfilter.Gain = sqrt(sps);

data = randi([0 3],200,1);
modData = qpsk(data);
txSig = txfilter(modData);

constDiagram = comm.ConstellationDiagram('SamplesPerSymbol',sps, ...
    'SymbolsToDisplaySource','Property','SymbolsToDisplay',100);

%scatterplot(txSig,sps)
constDiagram(txSig);