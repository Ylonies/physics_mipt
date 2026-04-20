from scipy.integrate import solve_ivp

from .models import InputHandler, PhysicsModels, ResultAnalyzer, ResultVisualizer

if __name__ == "__main__":
    params = InputHandler.get_parameters()

    modeler = PhysicsModels()
    t, theta, omega = modeler.solve_symplectic(params)

    analyzer = ResultAnalyzer()
    results = analyzer.analyze(t, theta, omega, params)

    visualizer = ResultVisualizer()
    visualizer.plot(results)
