import warnings

import numpy as np

from pulser import Pulse, Register, Sequence
from pulser.devices import Chadoq2
from pulser.waveforms import BlackmanWaveform, RampWaveform

warnings.filterwarnings("ignore")


# =============================================================================
# ====                              ANTI FERRO                             ====
# =============================================================================


# Parameters in rad/µs and ns
Omega_max = 2.3 * 2 * np.pi
U = Omega_max / 2.3

delta_0 = -6 * U
delta_f = 2 * U

t_rise = 252
t_fall = 500
t_sweep = (delta_f - delta_0) / (2 * np.pi * 10) * 1000

R_interatomic = Chadoq2.rydberg_blockade_radius(U)

N_side = 3
reg = Register.square(N_side, R_interatomic, prefix="q")


rise = Pulse.ConstantDetuning(
    RampWaveform(t_rise, 0.0, Omega_max), delta_0, 0.0
)
sweep = Pulse.ConstantAmplitude(
    Omega_max, RampWaveform(t_sweep, delta_0, delta_f), 0.0
)
fall = Pulse.ConstantDetuning(
    RampWaveform(t_fall, Omega_max, 0.0), delta_f, 0.0
)

seq = Sequence(reg, Chadoq2)
seq.declare_channel("ising", "rydberg_global")

seq.add(rise, "ising")
seq.add(sweep, "ising")
seq.add(fall, "ising")

print("=" * 80)
print("Antiferromagnetic state")
print()
print(seq.abstract_repr())
print()

# fmt: off
# import pdb; pdb.set_trace()
# fmt: on

# =============================================================================
# ====                   ANTI FERRO + optimal control                      ====
# =============================================================================

# This sequence has been created using optimal control.
# If you want to know more on the process, you can check
# https://pulser.readthedocs.io/en/stable/tutorials/optimization.html

seq = Sequence.deserialize(
    '{"_build": true, "__module__": "pulser.sequence", "__name__": "Sequence", "__args__": [{"_build": true, "__module__": "pulser.register.register", "__name__": "Register", "__args__": [{"q0": [-10.468682577131819, -10.468682577131819], "q1": [-3.48956085904394, -10.468682577131819], "q2": [3.489560859043939, -10.468682577131819], "q3": [10.468682577131819, -10.468682577131819], "q4": [10.468682577131819, -3.48956085904394], "q5": [10.468682577131819, 3.489560859043939], "q6": [10.468682577131819, 10.468682577131819], "q7": [3.489560859043939, 10.468682577131819], "q8": [-3.48956085904394, 10.468682577131819], "q9": [-10.468682577131819, 10.468682577131819], "q10": [-10.468682577131819, 3.489560859043939], "q11": [-10.468682577131819, -3.48956085904394]}], "__kwargs__": {}}, {"_build": false, "__module__": "pulser.devices", "__name__": "Chadoq2"}], "__kwargs__": {}, "__version__": "0.5.1.dev", "calls": [["declare_channel", ["ising", "rydberg_global"], {"initial_target": null}], ["add", [{"_build": true, "__module__": "pulser.pulse", "__name__": "Pulse", "__args__": [{"_build": true, "__module__": "pulser.waveforms", "__name__": "InterpolatedWaveform", "__args__": [1000, {"_build": true, "__module__": "numpy", "__name__": "array", "__args__": [[1e-09, 11.405902685751728, 15.707947559985698, 15.707947559985698, 1e-09]], "__kwargs__": {}}], "__kwargs__": {"times": null, "interpolator": "PchipInterpolator"}}, {"_build": true, "__module__": "pulser.waveforms", "__name__": "InterpolatedWaveform", "__args__": [1000, {"_build": true, "__module__": "numpy", "__name__": "array", "__args__": [[-31.41592653589793, -31.41592653589793, 8.854273919304688, 28.115521932222876, 31.41592653589793]], "__kwargs__": {}}], "__kwargs__": {"times": null, "interpolator": "PchipInterpolator"}}, 0.0], "__kwargs__": {"post_phase_shift": 0.0}}, "ising"], {}]], "vars": {}, "to_build_calls": []}'
)

print("=" * 80)
print("Antiferromagnetic state with optimal control")
print()
print(seq.abstract_repr())
print()

# =============================================================================
# ====                        CZ + Bell state prep                         ====
# =============================================================================

qubits = {"control": (-2, 0), "target": (2, 0)}
reg = Register(qubits)

seq = Sequence(reg, Chadoq2)
seq.declare_channel("digital", "raman_local", initial_target="control")
seq.declare_channel("rydberg", "rydberg_local", initial_target="control")

half_pi_wf = BlackmanWaveform(200, np.pi / 2)

ry = Pulse.ConstantDetuning(amplitude=half_pi_wf, detuning=0, phase=-np.pi / 2)
ry_dag = Pulse.ConstantDetuning(
    amplitude=half_pi_wf, detuning=0, phase=np.pi / 2
)

seq.add(ry, "digital")
seq.target("target", "digital")
seq.add(ry_dag, "digital")

pi_wf = BlackmanWaveform(200, np.pi)
pi_pulse = Pulse.ConstantDetuning(pi_wf, 0, 0)

max_val = Chadoq2.rabi_from_blockade(8)
two_pi_wf = BlackmanWaveform.from_max_val(max_val, 2 * np.pi)
two_pi_pulse = Pulse.ConstantDetuning(two_pi_wf, 0, 0)

seq.align("digital", "rydberg")
seq.add(pi_pulse, "rydberg")
seq.target("target", "rydberg")
seq.add(two_pi_pulse, "rydberg")
seq.target("control", "rydberg")
seq.add(pi_pulse, "rydberg")

seq.align("digital", "rydberg")
seq.add(ry, "digital")
seq.measure("digital")

print("=" * 80)
print("CZ + Bell state preparation")
print()
print(seq.abstract_repr())
print()
