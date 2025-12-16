class SimulationConfig:
    Lx = 1.0
    Ly = 1.0

    radius = 0.002        
    m = 1.0
    kB = 1.0

    dt = 1e-4               
    cell_size = 0.01       
    Mp = 500.0

    # 2D gas: only translational degrees of freedom.
    dof = 2
    gamma = (dof + 2) / dof