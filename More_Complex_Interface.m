clear; clc;

% Physical parameters
rho = 1.0;    % density
cp  = 1.0;    % specific heat
kappa = 1.0;  % thermal conductivity
Latent = 1.0; % latent heat per unit mass
T0  = 1.0;    % hot boundary temperature
Tm  = 0.0;    % melting temperature

max_iter = 20; % Maximum Newton iterations

% Domain and mesh
Lx = 1.0; Ly = 1.0;
Nx = 50; Ny = 50;

% Create structured triangular mesh
[X, Y] = meshgrid(linspace(0, Lx, Nx), linspace(0, Ly, Ny));
X = X(:); Y = Y(:);

% Triangulation (each quad divided into 2 triangles)
tri = delaunay(X, Y);
Nnodes = length(X);
Nelems = size(tri, 1);

% Identify boundary nodes
tol = 1e-10;
left_nodes   = find(abs(X) < tol);
right_nodes  = find(abs(X - Lx) < tol);
bottom_nodes = find(abs(Y) < tol);
top_nodes    = find(abs(Y - Ly) < tol);

% Initial Conditions: Circular Interface
center = [0.5, 0.5];
radius = 0.3;
dist = sqrt((X - center(1)).^2 + (Y - center(2)).^2);

% Define Solid (Inside) and Liquid (Outside)
solid_mask = (dist <= radius);
liquid_mask = ~solid_mask;

% Initialize Fields
T = zeros(Nnodes, 1);
H = zeros(Nnodes, 1);

% Solid Initialization
T(solid_mask) = Tm;
H(solid_mask) = cp * Tm;

% Liquid Initialization
T(liquid_mask) = T0;
H(liquid_mask) = cp * T0 + Latent;

% Boundary Conditions (Fix all walls to T0 to drive melting)
% Identify all boundary nodes
bnd_nodes = [left_nodes; right_nodes; bottom_nodes; top_nodes];
bnd_nodes = unique(bnd_nodes);

% Pre-allocate triplet arrays
max_nz = Nelems * 9; % 3x3 local matrix per element
I_K = zeros(max_nz, 1); J_K = zeros(max_nz, 1); V_K = zeros(max_nz, 1);
I_M = zeros(max_nz, 1); J_M = zeros(max_nz, 1); V_M = zeros(max_nz, 1);
idx = 0;

for e = 1:Nelems
    nodes = tri(e, :);
    x_e = X(nodes);
    y_e = Y(nodes);
    
    % Element area
    area = 0.5 * abs(det([1, x_e(1), y_e(1);
                          1, x_e(2), y_e(2);
                          1, x_e(3), y_e(3)]));
    
    % Gradients of shape functions
    grad_b = [y_e(2) - y_e(3); y_e(3) - y_e(1); y_e(1) - y_e(2)] / (2*area);
    grad_c = [x_e(3) - x_e(2); x_e(1) - x_e(3); x_e(2) - x_e(1)] / (2*area);
    
    % Local mass matrix (consistent)
    M_local = (area / 12) * [2 1 1; 1 2 1; 1 1 2];
    
    % Local stiffness matrix
    K_local = area * (grad_b*grad_b' + grad_c*grad_c');
    
    % Vectorized stacking
    for loc_i = 1:3
        for loc_j = 1:3
            idx = idx + 1;
            I_K(idx) = nodes(loc_i);
            J_K(idx) = nodes(loc_j);
            V_K(idx) = K_local(loc_i, loc_j);
            
            I_M(idx) = nodes(loc_i);
            J_M(idx) = nodes(loc_j);
            V_M(idx) = M_local(loc_i, loc_j);
        end
    end
end

% Assemble sparse matrices
K = sparse(I_K, J_K, V_K, Nnodes, Nnodes);
M = sparse(I_M, J_M, V_M, Nnodes, Nnodes);

% Time-stepping parameters
dt = 0.0001;  % Smaller time step for stability
t_final = 0.02;
steps = floor(t_final / dt);

% Identify free DOFs (all except Dirichlet boundary)
free_dofs = setdiff(1:Nnodes, bnd_nodes);

% Lumped mass matrix for explicit scheme (more stable)
M_lumped = spdiags(sum(M, 2), 0, Nnodes, Nnodes);

inv_rho = 1 / rho;

% Time integration
for n = 1:steps
    H_old = H;

    % Newton-Raphson iteration for nonlinear solve
    for iter = 1:max_iter
        % 1. Calculate T from current guess of H (using phase change relation)
        T_guess = zeros(Nnodes, 1);
        dTdH = zeros(Nnodes, 1); % Derivative for Jacobian

        % Solid (H <= cp*Tm)
        mask_solid = (H <= cp*Tm);
        T_guess(mask_solid) = H(mask_solid) / cp;
        dTdH(mask_solid) = 1/cp;
        
        % Mushy (cp*Tm < H < cp*Tm + Latent)
        mask_mushy = (H > cp*Tm) & (H < cp*Tm + Latent);
        T_guess(mask_mushy) = Tm;
        dTdH(mask_mushy) = 0; % T doesn't change with H here
        
        % Liquid (H >= cp*Tm + Latent)
        mask_liquid = (H >= cp*Tm + Latent);
        T_guess(mask_liquid) = Tm + (H(mask_liquid) - cp*Tm - Latent) / cp;
        dTdH(mask_liquid) = 1/cp;

        % 2. Enforce Dirichlet BCs on T_guess
        T_guess(bnd_nodes) = T0;
        H(bnd_nodes) = cp*T0 + Latent;

        % 3. Calculate Residual
        Res = M * (H - H_old) + dt * (kappa * inv_rho) * (K * T_guess);
        Res(bnd_nodes) = 0; % Enforce BCs in residual

        % 4. Check convergence
        res_norm = norm(Res(free_dofs)); % Only check free DOFs
        if res_norm < tol
            break;
        end

        % 5. Assemble Jacobian
        K_tangent = K * spdiags(dTdH, 0, Nnodes, Nnodes);
        Jac = M + dt * (kappa * inv_rho) * K_tangent;

        delta_H = zeros(Nnodes, 1);
        delta_H(free_dofs) = -Jac(free_dofs, free_dofs) \ Res(free_dofs);

        % 6. Update H
        H = H + delta_H;
    end

    if iter == max_iter
        warning('Newton did not converge at step %d', n);
    end

    solid = (H < cp*Tm + 1e-10);  % Small tolerance
    liquid = (H > cp*Tm + Latent - 1e-10);
    mushy = ~solid & ~liquid;
    
    T(solid) = H(solid) / cp;
    T(mushy) = Tm;  % During phase change, T stays at Tm
    T(liquid) = Tm + (H(liquid) - cp*Tm - Latent) / cp;
    
    % Check for NaN
    if any(isnan(T))
        fprintf('NaN detected at step %d\n', n);
        break;
    end
    
    % Progress indicator
    if mod(n, 100) == 0
        fprintf('Step %d/%d, T range: [%.3f, %.3f]\n', n, steps, min(T), max(T));
    end
end

% Plot final temperature field
figure;
trisurf(tri, X, Y, T, 'EdgeColor', 'none');
view(2);
xlabel('x'); ylabel('y');
title('Temperature Field at Final Time (FEM in Space)');
colorbar;
shading interp;
axis equal tight;