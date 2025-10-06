import math

class PhysicsModels:
    def __init__(self, g=9.81):
        self.g = g
    
    def viscous_friction_model(self, params):
        gamma = params['gamma']
        m = params['m']
        g = self.g
        
        def model(t, state):
            x, y, vx, vy = state
            
            Fx = -gamma * vx
            Fy = -gamma * vy
            
            dx_dt = vx
            dy_dt = vy
            dvx_dt = - (gamma / m) * vx
            dvy_dt = - g - (gamma / m) * vy 
            
            return [dx_dt, dy_dt, dvx_dt, dvy_dt]
        
        return model, "Вязкое трение"

    def quadratic_resistance_model(self, params):
        gamma = params['gamma']
        m = params['m']
        g = self.g
        
        def model(t, state):
            x, y, vx, vy = state
            
            v = math.sqrt(vx**2 + vy**2)
            
            dx_dt = vx
            dy_dt = vy
            dvx_dt = - (gamma / m) * abs(vx) * vx
            dvy_dt = - g - (gamma / m) * abs(vy) * vy
            
            return [dx_dt, dy_dt, dvx_dt, dvy_dt]
        
        return model, "Лобовое сопротивление"

    def get_model(self, params):
        if params['model_choice'] == "1":
            return self.viscous_friction_model(params)
        else:
            return self.quadratic_resistance_model(params)