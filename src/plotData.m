%From http://archive.ics.uci.edu/ml/datasets/Glass+Identification
%
%Attribute Information:
%
%1. Id number: 1 to 214
%2. RI: refractive index
%3. Na: Sodium
%4. Mg: Magnesium
%5. Al: Aluminum
%6. Si: Silicon
%7. K: Potassium
%8. Ca: Calcium
%9. Ba: Barium
%10. Fe: Iron
%11. Type of glass: (class attribute)
%-- 1 building_windows_float_processed
%-- 2 building_windows_non_float_processed
%-- 3 vehicle_windows_float_processed
%-- 4 vehicle_windows_non_float_processed (none in this database)
%-- 5 containers
%-- 6 tableware
%-- 7 headlamps
%
% trainIndex == trainIDs, similarly testIndex == testIDs.

% data contains:
% trainIndex testIndex
% trainData testData
% trainLabels testLabels
% trainIDs testIDs
data = load('../data/glass_dataset.mat');
all_data = [data.trainData; data.testData];
all_labels = [data.trainLabels; data.testLabels];

for i = min(all_labels):max(all_labels)
    if sum(all_labels==i) > 0
        figure(i);
        plotmatrix(all_data(all_labels==i, :));

        filename = sprintf('../img/correlation2d_%d.eps', i);
        %print(filename, '-depsc2');
    end
end

figure(max(all_labels)+1);
plotmatrix(all_data);
%print -depsc2 '../img/correlation2d_full.eps'
