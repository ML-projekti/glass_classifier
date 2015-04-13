%load data
data = load('../data/glass_dataset.mat');
train_data = [data.trainData; data.testData];


% Normalization - zero mean and unit variance in each dimension
[N,d]=size(train_data);
for i=1:d
    train_data(:,i)=(train_data(:,i)-mean(train_data(:,i)))/std(train_data(:,i));
end
size(train_data)


% %do PCA
% [COEFF,Y,latent] = princomp(train_data');

% run the CRP sampler to generate the posterior distribution over model 
% parameters
n_iter = 5000;
[class_id, phi, K_record, lP_record, alpha_record] = sampler(train_data', n_iter);

%do MDS and visualize
D=pdist(train_data);
D=squareform(D);
data_vis = mdscale(D, 2, 'Criterion', 'strain');

figure;
gscatter(data_vis(:,1),data_vis(:,2),class_id(:,n_iter));


%draw histogram of distribution of K over the latter half of the sample
sample = class_id(:,(n_iter/2)+1:n_iter);
for i=1:size(sample,2)
    sample_k(i) = size(unique(sample(:,i)),1);
end

figure;
k_dist = histogram(sample_k);


%visualize label distribution
series1 = [class_id(:,n_iter)];
series2 = [data.trainLabels; data.testLabels];
figure;
subplot(2,1,1)
h1 = histogram(series1);
subplot(2,1,2)
h2 = histogram(series2);

%visualize true labels
figure;
gscatter(data_vis(:,1),data_vis(:,2),series2);



