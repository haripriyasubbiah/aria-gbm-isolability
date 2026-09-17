export interface Citation {
  id: string;
  authors: string;
  title: string;
  venue: string;
  year: number;
  href: string;
  kind: "paper" | "preprint";
  role: string;
}

export const CITATIONS: Citation[] = [
  {
    id: "gavarini2022",
    authors: "Gavarini, Stucchi, Ruospo, Boracchi, Sánchez",
    title: "Open-Set Recognition: an Inexpensive Strategy to Increase DNN Reliability",
    venue: "IEEE IOLTS",
    year: 2022,
    href: "https://doi.org/10.1109/IOLTS56730.2022.9897805",
    kind: "paper",
    role: "Uses OSR/OOD scores as fault detectors — the confound this method refuses.",
  },
  {
    id: "progip2025",
    authors: "Joshi, So, Park, Ko, Jung, Ko, Hwang, Lee, Shrivastava",
    title: "ProGIP: Protecting Gradient-based Input Perturbation Approaches for OOD Detection From Soft Errors",
    venue: "ACM TECS",
    year: 2025,
    href: "https://doi.org/10.1145/3761796",
    kind: "paper",
    role: "Closest prior art: range checks separate ID / OOD / high-order bit-flips, but stay in activation magnitude space.",
  },
  {
    id: "drdna2024",
    authors: "Ma, Lin, Desmaison, Coburn, Moore, Sankar, Jiao",
    title: "Dr. DNA: Combating Silent Data Corruptions in Deep Learning using Distribution of Neuron Activations",
    venue: "ACM ASPLOS",
    year: 2024,
    href: "https://doi.org/10.1145/3620666.3651349",
    kind: "paper",
    role: "SDC signatures in neuron-activation histograms; OOD also moves those histograms.",
  },
  {
    id: "react2021",
    authors: "Sun, Guo, Li",
    title: "ReAct: Out-of-distribution Detection With Rectified Activations",
    venue: "NeurIPS",
    year: 2021,
    href: "https://proceedings.neurips.cc/paper/2021/hash/01894d6f048493d2cacde3c579c315a3-Abstract.html",
    kind: "paper",
    role: "Shows OOD inputs produce extreme activations — the same symptom high-order bit-flips produce.",
  },
  {
    id: "gertler1990",
    authors: "Gertler, Singer",
    title: "A new structural framework for parity equation-based failure detection and isolation",
    venue: "Automatica",
    year: 1990,
    href: "https://doi.org/10.1016/0005-1098(90)90133-3",
    kind: "paper",
    role: "Classical isolability via structured residuals. The dictionary here adds open-set novelty and NN execution.",
  },
  {
    id: "john2025",
    authors: "John et al.",
    title: "Multivariate Time Series Classification of Fermi-Detected Gamma-Ray Transients Using Convolutional-Recurrent Neural Networks",
    venue: "arXiv:2510.22412 / Mach. Learn.: Sci. Technol.",
    year: 2025,
    href: "https://arxiv.org/abs/2510.22412",
    kind: "preprint",
    role: "Public Fermi-GBM four-class classifier with outlier flagging; no execution-fault model.",
  },
  {
    id: "sirc2022",
    authors: "Xia, Bouganis",
    title: "Augmenting Softmax Information for Selective Classification with Out-of-Distribution Data",
    venue: "ACCV / IJCV",
    year: 2022,
    href: "https://doi.org/10.1007/978-3-031-20053-3_34",
    kind: "paper",
    role: "SCOD: reject ID errors and OOD. Causes remain input-side.",
  },
  {
    id: "narasimhan2024",
    authors: "Narasimhan, Menon, Rawat, Kumar, Agarwal",
    title: "Plugin Estimators for Selective Classification with Out-of-Distribution Detection",
    venue: "ICLR",
    year: 2024,
    href: "https://openreview.net/forum?id=OSlj6lfg1P",
    kind: "paper",
    role: "Theoretically grounded SCOD plug-in estimators; still accept/reject, no compute faults.",
  },
  {
    id: "ftcnn2020",
    authors: "Zhao, Wang, Lin, Li, Li",
    title: "FT-CNN: Algorithm-Based Fault Tolerance for Convolutional Neural Networks",
    venue: "IEEE TPDS",
    year: 2020,
    href: "https://doi.org/10.1109/TPDS.2020.2991788",
    kind: "paper",
    role: "ABFT checksums detect matmul errors independently of the input — used here as the execution residual, not as the contribution.",
  },
  {
    id: "goldstein2019",
    authors: "Goldstein et al.",
    title: "Evaluation of automated Fermi GBM localizations of gamma-ray bursts",
    venue: "ApJ",
    year: 2019,
    href: "https://doi.org/10.3847/1538-4357/ab22a5",
    kind: "paper",
    role: "Single-detector phosphorescent spikes are a real GBM artifact class — the sensor-fault proxy.",
  },
  {
    id: "shortgrb2025",
    authors: "Veske, Márka, Márka et al.",
    title: "A New Search Pipeline for Short Gamma-Ray Bursts in Fermi/GBM Data",
    venue: "ApJS / arXiv:2507.05739",
    year: 2025,
    href: "https://arxiv.org/abs/2507.05739",
    kind: "preprint",
    role: "Multi-detector matched filter plus single-detector vetoes; no neural execution integrity.",
  },
  {
    id: "bulgarelli2025",
    authors: "Bulgarelli et al.",
    title: "Onboard Machine Learning for High-Energy Observatories for Gamma-Ray Astronomy",
    venue: "Galaxies",
    year: 2025,
    href: "https://doi.org/10.3390/galaxies13020022",
    kind: "paper",
    role: "Onboard scientific ML for filtering and alerts, assuming correct execution.",
  },
];

export const GAP_CLAIMS = [
  "OOD / open-set scores live in the same activation-magnitude space as high-order silent data corruption (ReAct; Gavarini; Dr. DNA). A surprising logit cannot say whether the sky did something new or the matmul did.",
  "ProGIP is the honest closest paper: it does distinguish ID, OOD, and fault-affected GIP executions, but with range checks on gradients and OOD scores. Range is not an identifiability argument, and it does not use instrument physics.",
  "Classical structured residuals (Gertler) isolate sensor vs process vs actuator. They do not have a symbol for “the process is a new astrophysical class” or “the diagnostic computer is corrupted.”",
  "Fermi-GBM multi-detector rank-1 structure is already used to veto single-detector spikes. Nobody closes that residual with an input-invariant execution checksum to isolate discovery from silent corruption.",
] as const;
