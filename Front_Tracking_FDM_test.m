% 2D One-Phase Stefan Problem via front-tracking method
clear; clc; close all;

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
T(liquid_mask) = T0;

% Plot initial condition
figure;
contourf(X, Y, T, 20, 'EdgeColor', 'none'); hold on;
contour(X, Y, double(liquid_mask), [0.5 0.5], 'k--', 'LineWidth', 1);
xlabel('x'); ylabel('y');
title('Initial Temperature and Interface');
colorbar;

% Time-stepping parameters
dt      = 0.1 * min(dx, dy)^2 / k;  % stability limit
t_final = 0.1;
nt      = floor(t_final / dt);

% Main simulation loop of front-tracking
for n = 1:nt
    % Dirichlet at x=0 (first column)
    T(:,1) = T0;
    
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
    lapT(2:end-1,end) = (T(3:end,end) - 2*T(2:end-1,end) + T(1:end-2,end)) / dx^2 + (T(end,end) - 0.5*(T(end,end)+T(end,end))) / dy^2; % Neumann BC

    % Update temperature field using explicit Euler method
    T = T + dt * k * lapT;

    % Update interface via Stefan condition
    for j = 2:Ny-1
        % find cell index for interface location
        i = floor(interface(j)/dx) + 1;
        if i >= 1 && i < Nx
            % approximate normal derivative dT/dx
            dTdx = (T(j,i+1) - T(j,i)) / dx;
            % normal velocity Vn = -k/(L*rho) * dT/dx
            vn = - (k/(L*rho)) * dTdx;
            interface(j) = interface(j) + dt * vn;
        end
    end
    % enforce endpoints
    interface(1)  = interface(2);
    interface(Ny) = interface(Ny-1);

    % update liquid region mask
    liquid_mask = (X <= interface');

    % Update temperature in solid region
    T(~liquid_mask) = Tm; % solid region

end

% Plot final temperature field
figure;
contourf(X, Y, T, 20, 'EdgeColor', 'none'); hold on;
% Plot the final interface
plot(interface, y, 'r--', 'LineWidth', 1.5);
xlabel('x'); ylabel('y');
title('Temperature Field at Final Time by Front-Tracking Method');
colorbar;

