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
        ax = plt.gca()
        ax.set_axis_off()
        
        data = [
            {'title': 'ВРЕМЯ ПОЛЕТА', 'value': results['flight_time'], 'unit': 'с', 'color': '#FF6B6B'},
            {'title': 'ДАЛЬНОСТЬ', 'value': results['flight_distance'], 'unit': 'м', 'color': '#4ECDC4'},
            {'title': 'МАКС. ВЫСОТА', 'value': results['max_height'], 'unit': 'м', 'color': '#45B7D1'}
        ]
        
        plt.text(0.5, 0.9, 'ХАРАКТЕРИСТИКИ ПОЛЕТА', 
                transform=ax.transAxes, fontsize=14, fontweight='bold', 
                ha='center', va='center', color='#2C3E50')
        
        for i, item in enumerate(data):
            y_pos = 0.7 - i * 0.25
            
            plt.text(0.15, y_pos, item['title'], 
                    transform=ax.transAxes, fontsize=12, fontweight='bold',
                    ha='left', va='center', color=item['color'])
            
            plt.text(0.6, y_pos, f"{item['value']:.2f}", 
                    transform=ax.transAxes, fontsize=18, fontweight='bold',
                    color=item['color'], ha='center', va='center')
            
            plt.text(0.85, y_pos, item['unit'], 
                    transform=ax.transAxes, fontsize=12, fontweight='bold',
                    color=item['color'], ha='left', va='center')
            
            if i < len(data) - 1:
                plt.axhline(y=y_pos-0.15, color='#ECF0F1', linewidth=1)