%rng(123123123);

% Simulate data
theta_true = 4;
pi_true = 0.3;
n_samples = 10000;
z = (rand(n_samples,1) < pi_true) + 1; % 2 with probability pi_true
x = zeros(n_samples,1);
for i = 1:n_samples
    if z(i)==1
        x(i) = randn; % N(0,1)
    elseif z(i)==2
        x(i) = randn + theta_true;  % N(theta_true,1)
    end
end


% Parameters of the prior distributions.
alpha0 = 0.5;
beta0 = 0.2;

n_iter = 20;
% To keep track of the estimates of pi and theta in different iterations:
pi_est = zeros(n_iter,1);
th_est = zeros(n_iter,1);

% Some initial value for the things that will be updated
E_log_pi = -0.7;   % E(log(pi))
E_log_pi_c = -0.7;  % E(log(1-pi))
E_log_var = 4 .* ones(n_samples,1); % E((x_n-theta)^2)
r2 = 0.5 .* ones(n_samples,1); % Responsibilities of the second cluster.


for iter = 1:n_iter
    
    % Updated of responsibilites, factor q(z)
    log_rho1 = E_log_pi_c - 0.5 .* log(2*pi) - 0.5 .* (x.^2);
    log_rho2 = E_log_pi - 0.5 .* log(2*pi) - 0.5 .* E_log_var;
    max_log_rho = max(log_rho1, log_rho2); % Normalize to avoid numerical problems when exponentiating.
    rho1 = exp(log_rho1 - max_log_rho);
    rho2 = exp(log_rho2 - max_log_rho);
    r2 = rho2 ./ (rho1 + rho2);
    r1 = 1 - r2;
    
    N1 = sum(r1);
    N2 = sum(r2);
    
    % Update of factor q(pi)
    E_log_pi = psi(N2 + alpha0) - psi(N1 + N2 + 2*alpha0);
    E_log_pi_c = psi(N1 + alpha0) - psi(N1 + N2 + 2*alpha0);
    
    % Update of factor q(theta)
    x2_avg = 1/N2 .* sum(r2 .* x);
    beta_2 = beta0 + N2;
    m2 = 1/beta_2 .* N2 .* x2_avg;
    E_log_var = (x-m2).^2 + 1/beta_2;
    
    % Keep track of the current estimates
    pi_est(iter) = (N2 + alpha0) / (N1 + N2 + 2*alpha0);
    th_est(iter) = m2;    
end

disp(num2str([pi_est th_est]));
% With large n_samples, this should converge to the (pi_true, theta_true).
