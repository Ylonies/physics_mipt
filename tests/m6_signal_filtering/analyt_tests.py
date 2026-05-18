import numpy as np
import pytest

from src.m6_signal_filtering.models import integrator, simulation, spectrum, transfer

RC_R, RC_C = 1000.0, 1e-6
RLC_R, RLC_L, RLC_C = 10.0, 0.01, 1e-6
U0 = 1.0


@pytest.fixture
def rc_tau():
    return RC_R * RC_C


@pytest.fixture
def rc_wc(rc_tau):
    return 1.0 / rc_tau


@pytest.fixture
def rlc_w0():
    return transfer.rlc_omega_0(RLC_L, RLC_C)


@pytest.mark.parametrize(
    "omega, mag_exp, phase_deg",
    [
        (100, 0.995, -5.71),
        (500, 0.894, -26.6),
        (1000, 0.707, -45.0),
        (2000, 0.447, -63.4),
        (10000, 0.0995, -84.3),
    ],
)
def test_rc_transfer_function_table(omega, mag_exp, phase_deg):
    mag = float(transfer.rc_H_mag(omega, RC_R, RC_C))
    ph = float(np.degrees(transfer.rc_H_phase(omega, RC_R, RC_C)))
    assert abs(mag - mag_exp) / mag_exp < 0.002
    assert abs(ph - phase_deg) < 0.5


def test_rc_cutoff_minus_3db(rc_wc):
    mag = transfer.rc_H_mag(rc_wc, RC_R, RC_C)
    ph = np.degrees(transfer.rc_H_phase(rc_wc, RC_R, RC_C))
    assert abs(mag - 1 / np.sqrt(2)) < 1e-6
    assert abs(ph + 45) < 0.1


def test_rc_step_response(rc_tau):
    u0 = 1.0
    t_max, dt = 5 * rc_tau, rc_tau / 200
    hist = integrator.integrate_rc(lambda t: np.full_like(np.asarray(t), u0), RC_R, RC_C, 0.0, 0.0, t_max, dt)
    t = hist["t"]
    u_out = hist["u_out"]
    for frac in [0.5, 1.0, 2.0]:
        idx = int(np.argmin(np.abs(t - frac * rc_tau)))
        assert abs(u_out[idx] - u0 * (1 - np.exp(-frac))) / u0 < 0.01


@pytest.mark.parametrize("ratio", [0.1, 0.5, 1.0, 2.0, 10.0])
def test_rc_harmonic_steady_state(ratio, rc_wc, rc_tau):
    omega = ratio * rc_wc
    dt = min(rc_tau / 500, 2 * np.pi / (80 * omega))
    h = simulation.run_rc_harmonic(RC_R, RC_C, U0, omega, 0.05, dt)
    rel_a = abs(h["amp_num"] - h["amp_theor"]) / h["amp_theor"]
    dphi = abs(np.degrees(h["phase_num"] - h["phase_theor"]))
    assert rel_a < 0.01, f"δA={rel_a:.4f}"
    assert dphi < 1.0 or ratio >= 10.0 and dphi < 2.0


def test_rc_square_spectrum_vs_analytic(rc_tau):
    period = 0.01
    dt = spectrum.suggest_dt(rc_tau, 2 * np.pi / period)
    sq = simulation.run_rc_square(RC_R, RC_C, U0, period, 0.1, dt, 31)
    rel = np.abs(sq["amps_fft"] - sq["amps_ana"]) / (sq["amps_ana"] + 1e-12)
    assert np.max(rel[:6]) < 0.12


def test_rc_square_time_synthesis(rc_tau):
    period = 0.01
    dt = spectrum.suggest_dt(rc_tau, 2 * np.pi / period)
    sq = simulation.run_rc_square(RC_R, RC_C, U0, period, 0.1, dt, 31)
    n = int(round(period / dt))
    err = np.sqrt(np.mean((sq["u_out"][-n:] - sq["u_syn"][-n:]) ** 2))
    assert err < 0.05


def test_rlc_resonance_peak(rlc_w0):
    mag = float(transfer.rlc_H_mag(rlc_w0, RLC_R, RLC_L, RLC_C))
    ph = float(transfer.rlc_H_phase(rlc_w0, RLC_R, RLC_L, RLC_C))
    assert abs(mag - 1.0) < 1e-6
    assert abs(ph) < 1e-6


@pytest.mark.parametrize("ratio", [0.9, 1.1])
def test_rlc_off_resonance_gain(ratio, rlc_w0):
    mag = float(transfer.rlc_H_mag(ratio * rlc_w0, RLC_R, RLC_L, RLC_C))
    assert 0.35 < mag < 0.55


def test_rlc_quality_factor():
    assert abs(transfer.rlc_Q(RLC_R, RLC_L, RLC_C) - 10.0) < 0.01


@pytest.mark.parametrize("ratio", [0.5, 0.8, 1.0, 1.2, 1.5])
def test_rlc_harmonic_steady_state(ratio, rlc_w0):
    omega = ratio * rlc_w0
    q = transfer.rlc_Q(RLC_R, RLC_L, RLC_C)
    dt = spectrum.suggest_dt(2 * q / rlc_w0, 1.5 * rlc_w0)
    h = simulation.run_rlc_harmonic(RLC_R, RLC_L, RLC_C, U0, omega, 0.02, dt)
    rel_a = abs(h["amp_num"] - h["amp_theor"]) / (h["amp_theor"] + 1e-30)
    dphi = abs(np.degrees(h["phase_num"] - h["phase_theor"]))
    assert rel_a < 0.02
    assert dphi < 2.0


def test_rlc_square_spectrum(rlc_w0):
    period = 2 * np.pi / rlc_w0
    q = transfer.rlc_Q(RLC_R, RLC_L, RLC_C)
    dt = spectrum.suggest_dt(2 * q / rlc_w0, 2 * np.pi / period) * 0.5
    sq = simulation.run_rlc_square(RLC_R, RLC_L, RLC_C, U0, period, 0.04, dt, 21)
    rel = np.abs(sq["amps_fft"] - sq["amps_ana"]) / (sq["amps_ana"] + 1e-12)
    assert np.max(rel[:5]) < 0.15


def test_rlc_zero_initial():
    omega = transfer.rlc_omega_0(RLC_L, RLC_C)
    q = transfer.rlc_Q(RLC_R, RLC_L, RLC_C)
    dt = spectrum.suggest_dt(2 * q / omega, omega)
    hist = integrator.integrate_rlc(
        lambda t: U0 * np.sin(omega * np.asarray(t)), RLC_R, RLC_L, RLC_C, 0.0, 0.0, 0.0, 0.002, dt
    )
    assert abs(hist["u_out"][0]) < 1e-15
