data = load('../data/glass_dataset.mat');
all_data = [data.trainData; data.testData];
all_labels = [data.trainLabels; data.testLabels];

component_counts = 2:10;
mixtures = zeros(max(component_counts),...
    1+max(all_labels)-min(all_labels),...
    numel(component_counts));
likelihoods = zeros(1+max(all_labels)-min(all_labels),...
    numel(component_counts));

for i = 1:numel(component_counts)
    H = component_counts(i);
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
            mixtures(1:H, label, i) = P;

            Y = data.testData(data.testLabels == label, :)';
            logpold = zeros(size(Y,2), H);

            for i = 1:H
                invSi = inv(S(:,:,i));
                logdetSi=logdet(2*pi*S(:,:,i));
                for n = 1:size(Y,2)
                    v = Y(:,n) - m(:,i);
                    logpold(n,i) =-0.5*v'*invSi*v - 0.5*logdetSi + log(P(i));
                end
            end

            logl=0;
            for n=1:size(Y,2)
                logl = logl + logsumexp(logpold(n,:),ones(1,H));
            end
            likelihoods(label, i) =  logl;
        end
    end
end