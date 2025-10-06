import numpy as np

class ResultAnalyzer:
    
    @staticmethod
    def analyze(solution, params):
        x = solution.y[0]
        y = solution.y[1]
        vx = solution.y[2]
        vy = solution.y[3]

        ground_indices = np.where(y < 0)[0]
        if len(ground_indices) > 0:
            ground_index = ground_indices[0]
            flight_time = solution.t[ground_index]
            flight_distance = x[ground_index]
            max_height = np.max(y[:ground_index]) if ground_index > 0 else 0
        else:
            valid_indices = np.where(y >= 0)[0]
            
            if len(valid_indices) > 0:
                last_index = valid_indices[-1]
                flight_time = solution.t[last_index]
                flight_distance = x[last_index]
                max_height = np.max(y[:last_index+1])
            else:
                flight_time = solution.t[-1]
                flight_distance = x[-1]
                max_height = 0
        
        results = {
            'flight_time': flight_time,
            'flight_distance': flight_distance,
            'max_height': max_height,
            'trajectory': (x, y),
            'velocity': np.sqrt(vx**2 + vy**2),
            'velocity_x': vx,
            'velocity_y': vy,
            'time': solution.t
        }
        
        return results