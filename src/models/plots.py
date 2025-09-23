import matplotlib.pyplot as plt
import numpy as np

class ResultVisualizer:
    
    @staticmethod
    def plot(results, params, model_name):
        x, y = results['trajectory']
        
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 2, (1, 2))
        ResultVisualizer._plot_trajectory(x, y, model_name, params)
        
        plt.subplot(2, 2, 3)
        ResultVisualizer._plot_velocity(results['time'], results['velocity'])

        plt.subplot(2, 2, 4)
        ResultVisualizer._plot_parameters(results)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def _plot_trajectory(x, y, model_name, params):
        plt.plot(x, y, 'b-', linewidth=2, label='Траектория')
        
        plt.xlabel('Расстояние, м')
        plt.ylabel('Высота, м')
        plt.title(f'Траектория полета ({model_name})\n'
                 f'v₀={params["v0"]} м/с, α={np.degrees(params["alpha"]):.1f}°, γ={params["gamma"]}')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
    
    @staticmethod
    def _plot_velocity(time, velocity):
        plt.plot(time, velocity, 'r-', linewidth=2)
        plt.xlabel('Время, с')
        plt.ylabel('Скорость, м/с')
        plt.title('Зависимость скорости от времени')
        plt.grid(True, alpha=0.3)
        
        max_v = np.max(velocity)
        min_v = np.min(velocity)
        plt.annotate(f'Max: {max_v:.1f} м/с', xy=(0.05, 0.95), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        plt.annotate(f'Min: {min_v:.1f} м/с', xy=(0.05, 0.85), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    @staticmethod
    def _plot_parameters(results):
        parameters = ['Время полета', 'Дальность', 'Макс. высота']
        values = [results['flight_time'], results['flight_distance'], results['max_height']]
        units = ['с', 'м', 'м']
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        
        y_pos = np.arange(len(parameters))
        bars = plt.barh(y_pos, values, color=colors, edgecolor='black', alpha=0.8)
        
        plt.xlabel('Значение')
        plt.title('Характеристики полета')
        plt.yticks(y_pos, [f'{p}\n({u})' for p, u in zip(parameters, units)])

        for bar, value in zip(bars, values):
            width = bar.get_width()
            plt.text(width + max(values)*0.01, bar.get_y() + bar.get_height()/2,
                    f'{value:.2f}', va='center', fontweight='bold')
        
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()