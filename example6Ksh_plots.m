clear all;

load('dispWorkspace6');

%% Plot True Labels

labels = imgStore.Labels;
trueLabels = zeros(251, 251);
for i = 1:251
        for j = 1:251
                if i == j
                        continue
                end
                if labels(i) == labels(j)
                        trueLabels(i, j) = 1;
                end
        end
end

%% Plot divLabels

divs = zeros(251, 251);
for i = 1:251
    divs(i, :) = min([divVals(i, :); divVals(:, i)']);
end
accuracy = 0;
divLabels = zeros(251, 251);
for i = 1:251
        [vals, idxs] = mink(divs(i, :), 2);
        divLabels(i, idxs(2)) = 1;
        if labels(idxs(2)) == labels(i)
                accuracy = accuracy + 1;
        end
end

%% Save the 3 Plots

saveas(imagesc(divs), 'divs_test.png');
saveas(imagesc(trueLabels), 'trueLabels_test.png');
saveas(imagesc(divLabels), 'divLabels_test.png');
disp(accuracy);
