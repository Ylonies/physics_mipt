from .solver import CollisionSolver

class PhysicsModels:
    def __init__(self):
        self.solver = CollisionSolver()
    
    def elastic_wall_collision(self, params):
        results = self.solver.solve_elastic_wall(params)
        return results, "Абсолютно упругое столкновение со стенкой"

    def elastic_ball_collision(self, params):
        results = self.solver.solve_elastic_balls(params)
        return results, "Абсолютно упругое столкновение двух шаров"

    def hooke_wall_collision(self, params):
        results = self.solver.solve_hooke_wall(params)
        return results, "Модель упругого столкновения шара со стенкой (закон Гука)"

    def hooke_ball_collision(self, params):
        results = self.solver.solve_hooke_balls(params)
        return results, "Модель упругого столкновения двух шаров (закон Гука)"

    def get_model(self, params):
        object_choice = params['object_choice']
        model_choice = params['model_choice']
        
        if object_choice == '1':
            if model_choice == '1':
                return self.elastic_wall_collision(params)
            else:
                return self.hooke_wall_collision(params)
        else:
            if model_choice == '1':
                return self.elastic_ball_collision(params)
            else:
                return self.hooke_ball_collision(params)