from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.m6_signal_filtering.models import transfer


def _style_ax(ax):
    ax.grid(True, alpha=0.35, linestyle="--")
    ax.tick_params(labelsize=9)


def build_rc_figures(R: float, C: float, sweep: list[dict], square: dict, u0: float, period: float) -> tuple[plt.Figure, plt.Figure]:
    wc = transfer.rc_omega_c(R, C)
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.0, 1.0], hspace=0.42, wspace=0.32, left=0.07, right=0.97, top=0.92, bottom=0.08)

    ax_mag = fig.add_subplot(gs[0, 0])
    w = np.logspace(np.log10(0.01 * wc), np.log10(100 * wc), 400)
    ax_mag.semilogx(w / wc, 20 * np.log10(transfer.rc_H_mag(w, R, C)), "C0", lw=1.5)
    ax_mag.axhline(-3, ls=":", c="gray")
    ax_mag.axvline(1, ls=":", c="gray")
    ax_mag.set_ylabel("|H|, dB")
    ax_mag.set_title("АЧХ (теория)")
    _style_ax(ax_mag)

    ax_ph = fig.add_subplot(gs[0, 1])
    ax_ph.semilogx(w / wc, np.degrees(transfer.rc_H_phase(w, R, C)), "C1", lw=1.5)
    ax_ph.axvline(1, ls=":", c="gray")
    ax_ph.axhline(-45, ls=":", c="gray")
    ax_ph.set_ylabel("φ, °")
    ax_ph.set_xlabel(r"$\omega / \omega_c$")
    ax_ph.set_title("ФЧХ (теория)")
    _style_ax(ax_ph)

    for col, ratio in enumerate((0.1, 1.0)):
        h = next(x for x in sweep if abs(x["omega"] / wc - ratio) < 1e-6)
        ax = fig.add_subplot(gs[1, col])
        ax.plot(h["t"] * 1e3, h["u_in"], alpha=0.55, label="вход")
        ax.plot(h["t"] * 1e3, h["u_out"], label="выход")
        ax.set_xlabel("t, мс")
        ax.set_ylabel("U, В")
        ax.set_title(rf"Гармоника: $\omega/\omega_c={ratio}$")
        ax.legend(fontsize=8)
        _style_ax(ax)

    ratios = [h["omega"] / wc for h in sweep]
    ax_a = fig.add_subplot(gs[2, 0])
    ax_a.plot(ratios, [h["amp_num"] / u0 for h in sweep], "o-", label="числ.")
    ax_a.plot(ratios, [h["amp_theor"] / u0 for h in sweep], "x--", label="теор.")
    ax_a.set_xlabel(r"$\omega / \omega_c$")
    ax_a.set_ylabel(r"$A_{\mathrm{out}}/U_0$")
    ax_a.set_title("Установившаяся амплитуда")
    ax_a.legend(fontsize=8)
    _style_ax(ax_a)

    ax_p = fig.add_subplot(gs[2, 1])
    ax_p.plot(ratios, [np.degrees(h["phase_num"]) for h in sweep], "o-", label="числ.")
    ax_p.plot(ratios, [np.degrees(h["phase_theor"]) for h in sweep], "x--", label="теор.")
    ax_p.set_xlabel(r"$\omega / \omega_c$")
    ax_p.set_ylabel("φ, °")
    ax_p.set_title("Установившаяся фаза")
    ax_p.legend(fontsize=8)
    _style_ax(ax_p)
    fig.suptitle("RC: фильтр нижних частот", fontsize=14)

    fig_sq = plt.figure(figsize=(16, 5))
    gs2 = fig_sq.add_gridspec(1, 2, wspace=0.28, left=0.06, right=0.97, top=0.88, bottom=0.14)
    T = period
    t = square["t"]
    mask = (t >= t[-1] - 2 * T) & (t <= t[-1])
    ax_t = fig_sq.add_subplot(gs2[0, 0])
    ax_t.plot(t[mask] * 1e3, square["u_in"][mask], label="вход")
    ax_t.plot(t[mask] * 1e3, square["u_out"][mask], label="выход")
    ax_t.plot(t[mask] * 1e3, square["u_syn"][mask], "--", label="синтез", alpha=0.85)
    ax_t.set_xlabel("t, мс")
    ax_t.set_ylabel("U, В")
    ax_t.set_title("Меандр: установившийся режим")
    ax_t.legend(fontsize=8)
    _style_ax(ax_t)

    ax_s = fig_sq.add_subplot(gs2[0, 1])
    n = min(7, len(square["ks_fft"]))
    x = np.arange(n)
    bw = 0.35
    ax_s.bar(x - bw / 2, square["amps_fft"][:n], bw, label="числ.")
    ax_s.bar(x + bw / 2, square["amps_ana"][:n], bw, label="аналит.")
    ax_s.set_xticks(x)
    ax_s.set_xticklabels([str(int(k)) for k in square["ks_fft"][:n]])
    ax_s.set_xlabel("гармоника k")
    ax_s.set_ylabel("A_k, В")
    ax_s.set_title("Спектр выхода")
    ax_s.legend(fontsize=8)
    _style_ax(ax_s)
    fig_sq.suptitle("RC: меандр", fontsize=13)
    return fig, fig_sq


def build_rlc_figures(
    R: float, L: float, C: float, sweep: list[dict], square: dict | None, u0: float
) -> tuple[plt.Figure, plt.Figure | None]:
    w0 = transfer.rlc_omega_0(L, C)
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.0, 1.0], hspace=0.42, wspace=0.32, left=0.07, right=0.97, top=0.92, bottom=0.08)

    ax_mag = fig.add_subplot(gs[0, 0])
    w = np.logspace(np.log10(0.3 * w0), np.log10(3 * w0), 400)
    ax_mag.semilogx(w / w0, transfer.rlc_H_mag(w, R, L, C), "C0", lw=1.5)
    ax_mag.axvline(1, ls=":", c="gray")
    ax_mag.set_ylabel("|H|")
    ax_mag.set_title("АЧХ (теория)")
    _style_ax(ax_mag)

    ax_ph = fig.add_subplot(gs[0, 1])
    ax_ph.semilogx(w / w0, np.degrees(transfer.rlc_H_phase(w, R, L, C)), "C1", lw=1.5)
    ax_ph.axvline(1, ls=":", c="gray")
    ax_ph.set_ylabel("φ, °")
    ax_ph.set_xlabel(r"$\omega / \omega_0$")
    ax_ph.set_title("ФЧХ (теория)")
    _style_ax(ax_ph)

    for col, ratio in enumerate((0.8, 1.0)):
        h = next(x for x in sweep if abs(x["omega"] / w0 - ratio) < 1e-6)
        ax = fig.add_subplot(gs[1, col])
        ax.plot(h["t"] * 1e3, h["u_in"], alpha=0.55, label="вход")
        ax.plot(h["t"] * 1e3, h["u_out"], label="выход")
        ax.set_xlabel("t, мс")
        ax.set_ylabel("U, В")
        ax.set_title(rf"Гармоника: $\omega/\omega_0={ratio}$")
        ax.legend(fontsize=8)
        _style_ax(ax)

    ratios = [h["omega"] / w0 for h in sweep]
    ax_a = fig.add_subplot(gs[2, 0])
    ax_a.plot(ratios, [h["amp_num"] / u0 for h in sweep], "o-", label="числ.")
    ax_a.plot(ratios, [h["amp_theor"] / u0 for h in sweep], "x--", label="теор.")
    ax_a.set_xlabel(r"$\omega / \omega_0$")
    ax_a.set_ylabel(r"$A_{\mathrm{out}}/U_0$")
    ax_a.set_title("Установившаяся амплитуда")
    ax_a.legend(fontsize=8)
    _style_ax(ax_a)

    ax_p = fig.add_subplot(gs[2, 1])
    ax_p.plot(ratios, [np.degrees(h["phase_num"]) for h in sweep], "o-", label="числ.")
    ax_p.plot(ratios, [np.degrees(h["phase_theor"]) for h in sweep], "x--", label="теор.")
    ax_p.set_xlabel(r"$\omega / \omega_0$")
    ax_p.set_ylabel("φ, °")
    ax_p.set_title("Установившаяся фаза")
    ax_p.legend(fontsize=8)
    _style_ax(ax_p)
    fig.suptitle("RLC: полосовой фильтр (выход на R)", fontsize=14)

    fig_sq = None
    if square is not None:
        fig_sq = plt.figure(figsize=(16, 5))
        gs2 = fig_sq.add_gridspec(1, 2, wspace=0.28, left=0.06, right=0.97, top=0.88, bottom=0.14)
        T = 2 * np.pi / w0
        t = square["t"]
        mask = (t >= t[-1] - 2 * T) & (t <= t[-1])
        ax_t = fig_sq.add_subplot(gs2[0, 0])
        ax_t.plot(t[mask] * 1e3, square["u_in"][mask], label="вход")
        ax_t.plot(t[mask] * 1e3, square["u_out"][mask], label="выход")
        ax_t.plot(t[mask] * 1e3, square["u_syn"][mask], "--", label="синтез", alpha=0.85)
        ax_t.set_xlabel("t, мс")
        ax_t.set_title("Меандр (период ≈ 1/ω₀)")
        ax_t.legend(fontsize=8)
        _style_ax(ax_t)
        ax_s = fig_sq.add_subplot(gs2[0, 1])
        n = min(7, len(square["ks_fft"]))
        x = np.arange(n)
        bw = 0.35
        ax_s.bar(x - bw / 2, square["amps_fft"][:n], bw, label="числ.")
        ax_s.bar(x + bw / 2, square["amps_ana"][:n], bw, label="аналит.")
        ax_s.set_xticks(x)
        ax_s.set_xticklabels([str(int(k)) for k in square["ks_fft"][:n]])
        ax_s.set_title("Спектр выхода")
        ax_s.legend(fontsize=8)
        _style_ax(ax_s)
        fig_sq.suptitle("RLC: меандр", fontsize=13)
    return fig, fig_sq


def show_figures(*figs: plt.Figure | None):
    import matplotlib.pyplot as plt

    for f in figs:
        if f is not None:
            plt.figure(f.number)
    plt.show()


def save_figures(path: Path, prefix: str, main: plt.Figure, extra: plt.Figure | None):
    path.mkdir(parents=True, exist_ok=True)
    main.savefig(path / f"{prefix}_main.png", dpi=140, bbox_inches="tight")
    plt.close(main)
    if extra is not None:
        extra.savefig(path / f"{prefix}_square.png", dpi=140, bbox_inches="tight")
        plt.close(extra)
