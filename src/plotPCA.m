data = load('../data/glass_dataset.mat');
all_data = [data.trainData; data.testData];
all_labels = [data.trainLabels; data.testLabels];
num_labels = 1 + max(all_labels) - min(all_labels);

[COEFF,SCORE,LATENT,TSQUARE] = princomp(data.testData);