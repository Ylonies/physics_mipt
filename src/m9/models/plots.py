import matplotlib.pyplot as plt


class ResultVisualizer:
    @staticmethod
    def _draw_lenses(ax, thin_lenses, image_plane):
        for lens in thin_lenses:
            ax.axvline(lens.position, color="green", ls="--", lw=1.0, alpha=0.8)
        ax.axvline(image_plane, color="red", ls=":", lw=1.2, label="изображение")

    @staticmethod
    def plot(results, params, model_name: str) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)

        axes[0].imshow(
            results["image"],
            origin="lower",
            cmap="inferno",
            aspect="auto",
            extent=[-1, 1, -1, 1],
        )
        axes[0].set_title("Изображение объекта")

        axes[1].plot(results["y_axis"], results["profile"], lw=1.5)
        axes[1].set_title("Профиль яркости")
        axes[1].set_xlabel("y (норм.)")
        axes[1].grid(True, alpha=0.3)

        for path in results["ray_paths"][:35]:
            axes[2].plot([p[0] for p in path], [p[1] for p in path], "-", lw=0.7, alpha=0.45)
        ResultVisualizer._draw_lenses(axes[2], results["thin_lenses"], results["image_plane"])
        axes[2].set_title("Ход лучей")
        axes[2].set_xlabel("x, м")
        axes[2].set_ylabel("y, м")
        axes[2].legend(loc="upper right", fontsize=8)
        axes[2].grid(True, alpha=0.3)

        t_note = ""
        if params["lens_mode"] == "2":
            t_note = f", t={params['lens_thickness_m']*1000:.1f} мм"
        fig.suptitle(
            f"М9: {model_name}{t_note} | M(теор.)={results['m_theory']:.2f}, "
            f"M(лучи)={results['m_ray']:.2f}"
        )
        plt.show()
