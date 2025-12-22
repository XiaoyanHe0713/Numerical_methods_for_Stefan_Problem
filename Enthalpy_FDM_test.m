% 2D One-Phase Stefan Problem via Enthalpy Method
% MATLAB implementation

clear; clc;

% Physical parameters
rho = 1.0;    % density
c   = 1.0;    % specific heat
k   = 1.0;    % thermal conductivity
L   = 1.0;    % latent heat per unit mass
T0  = 1.0;    % hot boundary temperature
Tm  = 0.0;    % melting temperature

% Domain and discretization
Nx = 200; Ny = 200;
Lx = 1.0; Ly = 1.0;
dx = Lx/(Nx-1); dy = Ly/(Ny-1);
x  = linspace(0, Lx, Nx);
y  = linspace(0, Ly, Ny);
[X, Y] = meshgrid(x, y);

% Initial interface: x = 0.5 + 0.1*sin(2*pi*y)
interface = 0.5 + 0.1*sin(2*pi*y);
liquid_mask = (X <= interface');   % Ny-by-Nx logical array

% Initialize fields
T = Tm * ones(Ny, Nx);
H = c * Tm * ones(Ny, Nx);
T(liquid_mask) = T0;
H(liquid_mask) = c*T0 + L;

% Plot initial condition
figure;
contourf(X, Y, T, 20, 'EdgeColor', 'none'); hold on;
contour(X, Y, double(liquid_mask), [0.5 0.5], 'k--', 'LineWidth', 1);
xlabel('x'); ylabel('y');
title('Initial Temperature and Interface');
colorbar;

% Time-stepping parameters
dt = 0.1 * min(dx, dy)^2 / k;  % stability limit
steps = 1000;

% Main simulation loop
for n = 1:steps
    % Dirichlet at x=0 (first column)
    T(:,1) = T0;
    H(:,1) = c*T0 + L;

    % Compute Laplacian of T
    lapT = zeros(Ny, Nx);
    % interior nodes
    lapT(2:end-1,2:end-1) = (T(2:end-1,3:end) - 2*T(2:end-1,2:end-1) + T(2:end-1,1:end-2)) / dx^2 + ...
                             (T(3:end,2:end-1) - 2*T(2:end-1,2:end-1) + T(1:end-2,2:end-1)) / dy^2;
    % Neumann (insulated) on y-boundaries
    lapT(1,2:end-1)   = (T(2,2:end-1) - T(1,2:end-1)) / dy^2 + (T(1,3:end) - 2*T(1,2:end-1) + T(1,1:end-2)) / dx^2;
    lapT(end,2:end-1) = (T(end-1,2:end-1) - T(end,2:end-1)) / dy^2 + (T(end,3:end) - 2*T(end,2:end-1) + T(end,1:end-2)) / dx^2;
    % Neumann on x=Ly boundary edges
    lapT(2:end-1,1)   = (T(2:end-1,2) - T(2:end-1,1)) / dx^2 + (T(3:end,1) - 2*T(2:end-1,1) + T(1:end-2,1)) / dy^2;
    lapT(2:end-1,end) = (T(2:end-1,end-1) - T(2:end-1,end)) / dx^2 + (T(3:end,end) - 2*T(2:end-1,end) + T(1:end-2,end)) / dy^2;

    % Update enthalpy
    H = H + k * dt * lapT;

    % Update temperature from enthalpy
    solid  = (H <= L);
    liquid = ~solid;
    T(solid)  = Tm; % solid region
    T(liquid) = (H(liquid) - L) / c;
end

% Plot final temperature field
figure;
contourf(X, Y, T, 20, 'EdgeColor', 'none');
xlabel('x'); ylabel('y');
title('Temperature Field at Final Time by Enthalpy Method');
colorbar;
