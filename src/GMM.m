
%% @param   D   dataset D = {x_1, x_2, ..., x_n}, where every data element x_i
%%              is d-dimensional vector meansurement. Dataset contains N
%%              measurements, thus D is a Nxd matrix.
%% @param   K   Number of components in the mixture model.
%% @return      Returns the Gaussian density parameters of each K components.
function [mu, sigma] = GMM(D, K)
    N = size(D, 1);
    d = size(D, 2);
    mu = zeros(d, K);
    sigma = zeros(d, d, K);

    if N <= K % Validate sensible paratemer values.
        return
    end

    % Initialize means with random measurements and
    % components covariances with the covar of the entire data set.
    % TODO: make these into a parameter. It could be useful to try
    %       initializing the EM-algo with the output of k-means algo,
    %       i.e. for each cluster, the mean is the cluster center and
    %       covariance is the covar of the k-means produced cluster.
    mu = D(randperm(N, K), :)';
    sigma = repmat(cov(D), [1, 1, K]);

end