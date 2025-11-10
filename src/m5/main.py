from scipy.integrate import solve_ivp

from .models import InputHandler, PhysicsModels, ResultAnalyzer, ResultVisualizer

if __name__ == "__main__":
    params = InputHandler.get_parameters()
    modeler = PhysicsModels()
    model, y0, t_span, t_eval = modeler.ellipse_pendulum_model(params)
    solution = solve_ivp(model, t_span, y0, t_eval=t_eval, rtol=1e-8)
    analyzer = ResultAnalyzer()
    results = analyzer.analyze(solution, params)
    visualizer = ResultVisualizer()
    visualizer.plot(results)