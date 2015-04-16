data = load('../data/glass_dataset.mat');
all_data = [data.trainData; data.testData];
all_labels = [data.trainLabels; data.testLabels];
num_labels = 1 + max(all_labels) - min(all_labels);

cfsnmat_GMM_rel_sum = zeros(6,6,9);

for master_cycle = 1:10

component_counts = 2:10;
mixtures = zeros(max(component_counts),...
    num_labels,...
    numel(component_counts));
likelihoods = ones(size(data.testData, 1),...
    num_labels,...
    numel(component_counts)) * -123456;

for ii = 1:numel(component_counts)
    H = component_counts(ii);
    for label = min(all_labels):max(all_labels)
        X = data.trainData(data.trainLabels == label, :)';

        if numel(X) > 0 && size(X, 2) > H
            % EM training :
            opts.plotlik=0;
            opts.plotsolution=0;
            opts.maxit=200;
            opts.minChange=0.001;
            opts.minChangeCount=5;
            opts.minDeterminant=0.0001;
            [P,m,S,loglik,phgn]=GMMem(X,H,opts);
            mixtures(1:H, label, ii) = P;

            % Calculate the log likelihood of each data point for this GMM.
            Y = data.testData';
            logpold = zeros(size(Y,2), H);

            for i = 1:H
                invSi = inv(S(:,:,i));
                logdetSi=logdet(2*pi*S(:,:,i));
                for n = 1:size(Y,2)
                    v = Y(:,n) - m(:,i);
                    logpold(n,i) =-0.5*v'*invSi*v - 0.5*logdetSi + log(P(i));
                end
            end

            for n=1:size(Y,2)
                likelihoods(n, label, ii) = logsumexp(logpold(n,:),ones(1,H));
            end
        end
    end
end

predicted_labels = zeros(size(data.testData, 1), numel(component_counts));
for i = 1:numel(component_counts)
    for n = 1:size(data.testData, 1)
        predicted_labels(n, i) = find(...
            likelihoods(n, :, i) == max(likelihoods(n, :, i), [], 2));
    end
end

cfsnmat_GMM_abs = zeros(6, 6, size(predicted_labels, 2));
cfsnmat_GMM_rel = zeros(6, 6, size(predicted_labels, 2));

for i = 1:size(predicted_labels, 2)
    cfsnmat_GMM_abs(:, :, i) = confusionmat(...
        data.testLabels, predicted_labels(:, i));
    cfsnmat_GMM_rel(:, :, i) = cfsnmat_GMM_abs(:, :, i) ./ repmat(...
        sum(cfsnmat_GMM_abs(:, :, i), 2), [1, 6]);
end

cfsnmat_GMM_rel_sum = cfsnmat_GMM_rel_sum + cfsnmat_GMM_rel;
end % of master_cycle

cfsnmat_GMM_rel_sum = cfsnmat_GMM_rel_sum / 10;

model_kNN = fitcknn(data.trainData, data.trainLabels,...
    'Distance', 'euclidean', 'NumNeighbors', 1);
kNN_labels = predict(model_kNN, data.testData);

cfsnmat_kNN_abs = confusionmat(data.testLabels, kNN_labels);
cfsnmat_kNN_rel = cfsnmat_kNN_abs ./ repmat(sum(cfsnmat_kNN_abs, 2), [1, 6]);
