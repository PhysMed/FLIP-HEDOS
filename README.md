# FLIP-HEDOS 🩸☢️
FLow and Irradiation Personalized – HEmatological DOSe

## Installation

Checkout the python source code

```bash
$ git clone https://github.com/PhysMed/FLIP-HEDOS.git
```

Install dependent packages

```bash
$ pip install -r requirements.txt
```

## Clinical and Scientific Background
**FLIP-HEDOS** is a Python-based implementation that integrates the **FLIP** method (*FLow and Irradiation Personalized*) method
into the HEDOS hematological dose model. The original HEDOS framework simulates whole-body circulation using 32 interconnected,
standardized (population-based) compartments. FLIP-HEDOS adopts the modular architecture of HEDOS, allowing each anatomical configuration
to be modeled independently within a unified computational framework. By incorporating the FLIP method, specific vascular regions can be
individualized, resulting in the enhanced FLIP-HEDOS model.

This approach establishes an advanced dosimetric framework for estimating patient-specific blood dose during external beam
radiotherapy (EBRT). It evaluates the radiation exposure of circulating blood for both proton and photon radiotherapy modalities,
treating blood as a **dynamic** organ at risk (OAR).

Radiation-induced toxicity due to blood dose is a side effect of EBRT, strongly associated with poor disease outcome across multiple cancer types.

Quantifying this dose in FLIP-HEDOS requires tracking blood particles through two coupled systems simultaneously:

1. **Population-based compartments** — stochastic compartments based on ICRP Publication 89 reference anatomical and physiological data (Valentin 2002).
2. **Patient-specific compartments** — individual large blood vessels extracted from CT segmentation and blood flow velocity field measured with 3D phase-contrast MRI to generate Lagrangian blood trajectories.


### `BloodDose.py` — Entry Point

For blood dose calculations, the configuration parameters (patient, treatment and simulation parameters) are set up in `BloodDose`.
Interactive script that collects all simulation parameters via `input()` prompts and calls the main workflow function:
`BloodDoseFromDVHandPatientSpecific`.


The calculation of blood dose follows these steps in succession:

- `FlowModel`: Set up a graph that reflects the connectivity and magnitude of blood flow between compartments (ICRP Publication 89 physiology). Convert this into a matrix of transition probabilities. The FLIP extension inserts the patient-specific arterial and venous modules into this graph.
- `TemporalDistribution`: Simulate the blood flow over time. Blood particles flow through the model by a stochastic jumping process with Weibull-distributed transit times. When a blood particle enters a FLIP compartment, it follows a blood trajectory through the real vasculature.
- `CompartmentDose`: Accumulate dose in blood particles over time.
## Input data

All patient data must be preprocessed into `.mat` (MATLAB) files:

- **CT + TPS dose maps** — dose per energy layer (proton RT) or arc segment (photon RT)
- **PC-MRI** — 3D blood velocity field and vessel segmentation → Lagrangian blood trajectories
- **Temporal structure of the beam delivery** — real beam-on temporal sequence (BEX signal for proton RT; log for photon RT)
- **Organ DVH files** — per-organ dose-volume histograms from the TPS (`.csv`)

The whole-body compartmental model (ICRP 89 Excel tables) will be provided soon in `input/phantom/`.

Two trial patients: `Patient19` (thorax-abdomen, proton therapy) and `Patient20` (head-and-neck, proton therapy), will also be provided soon.

## Usage

```bash
python BloodDose.py
```

The script prompts interactively for patient, treatment and simulation parameters.

## Output

Results are written to `output/PatientXX/`:

- Visualizes final blood volume distribution across compartments.
- Computes dose received by blood particles in FLIP (patient-specific) compartments.
- Analyzes how often blood particles visit FLIP compartments.
- Generates DVHs for blood across organs. All organs are plotted in a single figure.
- Dose metrics summary: mean dose, percentage of blood particles receiving more than 0.001 Gy and 0.1 Gy.
- Accumulates dose over all treatment fractions.


## Team

- Marina García-Cardosa ([mgarciacard@unav.es](mailto:mgarciacard@unav.es))
- Chris Beekman ([cbeekman@mgh.harvard.edu](mailto:cbeekman@mgh.harvard.edu))
- Javier Burguete ([javier@unav.es](mailto:javier@unav.es))
- Harald Paganetti ([hpaganetti@mgh.harvard.edu](mailto:hpaganetti@mgh.harvard.edu))

## Publications

- **FLIP-HEDOS approach** García-Cardosa M, et al. FLIP-HEDOS: a patient-specific blood dose quantification model during radiotherapy treatments. *Phys. Med. Biol.* 2026; 71:095005. https://doi.org/10.1088/1361-6560/ae6015
- **Pure FLIP approach** García-Cardosa M, et al. FLIP: a novel method for patient-specific dose quantification in circulating blood in large
vessels during proton or photon external beam radiotherapy treatments. *Phys. Med. Biol.* 2024; 69:225017. https://doi.org/10.1088/1361-6560/ad8ea5
- **Second pure HEDOS approach** Beekman C, et al. A stochastic model of blood flow to calculate blood dose during radiotherapy. *Phys. Med. Biol.* 2023;
68:225007. https://doi.org/10.1088/1361-6560/ad02d6
- **First pure HEDOS approach** Shin J, et al. HEDOS — a computational tool to assess radiation dose to circulating blood cells during external beam
radiotherapy based on whole-body blood flow simulations. *Phys. Med. Biol.* 2021; 66:164001. https://doi.org/10.1088/1361-6560/ac16ea

## License
This project is licensed under the GNU General Public License v3.0 (GPLv3).

Copyright (C) 2026 PhysMed Research Group - University of Navarra

## Third-Party Code
This project includes code licensed under the MIT License.

- Copyright (c) 2021 MGH Radiation Oncology
- Copyright (c) 2023 MGH Radiation Oncology

The original MIT-licensed code remains under its original license.
