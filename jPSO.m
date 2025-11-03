function [sFeat, Sf, Nf, curve] = jPSO(feat, label, N, max_Iter, c1, c2, w, HO)
    % Parameters
    lb = 0;
    ub = 1; 
    thres = 0.5;

    % Objective function
    fun = @jFitnessFunction; 
    % Number of dimensions
    dim = size(feat, 2); 
    % Initial 
    X = zeros(N, dim);
    V = zeros(N, dim); 
    for i = 1:N
        for d = 1:dim
            X(i, d) = lb + (ub - lb) * rand();
        end
    end  

    % Fitness
    fit = zeros(1, N); 
    fitG = inf;
    for i = 1:N 
        fit(i) = fun(feat, label, (X(i, :) > thres), HO); 
        % Gbest update
        if fit(i) < fitG
            Xgb = X(i, :); 
            fitG = fit(i);
        end
    end

    % PBest
    Xpb = X; 
    fitP = fit;

    % Pre
    curve = inf;
    t = 1;  

    % Iterations
    while t <= max_Iter
        for i = 1:N
            % Define the neighborhood range for the ring topology
            left_neighbor = mod(i - 2, N) + 1; % Index of the left neighbor
            right_neighbor = mod(i, N) + 1;    % Index of the right neighbor
            
            for d = 1:dim
                r1 = rand();
                r2 = rand();
                % Velocity update 
                V(i, d) = w * V(i, d) + c1 * r1 * (Xpb(i, d) - X(i, d)) + ...
                    c2 * r2 * (Xpb(left_neighbor, d) + Xpb(right_neighbor, d) - 2 * X(i, d)); % Local best (ring) topology
                % Position update 
                X(i, d) = X(i, d) + V(i, d);
            end
            
            % Boundary
            X(i, :) = max(min(X(i, :), ub), lb);

            % Fitness
            fit(i) = fun(feat, label, (X(i, :) > thres), HO);
            
            % Pbest update
            if fit(i) < fitP(i)
                Xpb(i, :) = X(i, :); 
                fitP(i) = fit(i);
            end
            
            % Gbest update
            if fitP(i) < fitG
                Xgb = Xpb(i, :);
                fitG = fitP(i);
            end
        end
        
        curve(t) = fitG;
        fprintf('\nIteration %d GBest (PSO) = %f', t, curve(t))
        t = t + 1;
    end
    
    % Select features based on selected index
    Pos = 1:dim;
    Sf = Pos((Xgb > thres) == 1); 
    sFeat = feat(:, Sf); 
    Nf = length(Sf);
end
