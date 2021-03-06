d = [[130   2   0   0   9]
 [  0 232   1   0   0]
 [  0   2 153  18   1]
 [  0  20  18 303   0]
 [  2   5   2   2 250]];

score = sum(diag(d))/sum(sum(d));

dm = d/max(max(d))*100;

s = sum(d);
for i = [1:5]
    for j = [1:5]
        dm(i, j) = d(i, j)/s(j) * 100;
    end
end


h = heatmap(dm);

title('Magnitude Spectrogram Classification Accuracy')
xlabel('Predicted Class Accuracy/%')
ylabel('True Class Accuracy/%')
colorbar('off')