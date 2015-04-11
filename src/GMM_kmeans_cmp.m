% 40 cm valkokultakaulaketju koivurunkokorulle.
data = load('../data/glass_dataset.mat');
all_data = [data.trainData; data.testData];
all_labels = [data.trainLabels; data.testLabels];

component_counts = 2:10;
mixtures = zeros(max(component_counts),...
    1+max(all_labels)-min(all_labels),...
    numel(component_counts));
likelihoods = ones(size(data.testData, 1),...
    1+max(all_labels)-min(all_labels),...
    numel(component_counts)) * -123456;

for ii = 1:numel(component_counts)
    H = component_counts(ii);
    for label = min(all_labels):max(all_labels)
        X = data.trainData(data.trainLabels == label, :)';

        if numel(X) > 0 && size(X, 2) > H
            % EM training :
            opts.plotlik=1;
            opts.plotsolution=1;
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

predicted_classes = zeros(size(data.testData, 1), numel(component_counts));
for i = 1:numel(component_counts)
    for n = 1:size(data.testData, 1)
        predicted_classes(n, i) = find(...
            likelihoods(n, :, i) == max(likelihoods(n, :, i), [], 2));
    end
end
