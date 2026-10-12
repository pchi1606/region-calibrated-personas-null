"""Ground truth for s12: published INTRA per-country CPT estimates.

Values transcribed 2026-10-08 from the open-access table in Xie, Hwang &
Pantelous (2018, JBF; open-access copy at https://apjfs.org/file/download/6487?view=1), who report their per-country estimates as closely
matching the INTRA-published values of Wang, Rieger & Hens (2017) and
Rieger, Wang & Hens (2015). Columns: lambda (loss aversion), alpha (curvature),
delta (probability weighting), with published SEs.
"""
import math

# (country, lambda, se_l, alpha, se_a, delta, se_d)
RAW = [
    ("Australia", 1.651, 0.009, 1.210, 0.005, 0.771, 0.010),
    ("Austria", 1.819, 0.005, 1.158, 0.004, 0.792, 0.007),
    ("Belgium", 1.809, 0.006, 1.438, 0.004, 0.698, 0.010),
    ("Brazil", 0.603, 0.021, 1.494, 0.005, 0.807, 0.008),
    ("Canada", 1.940, 0.017, 1.459, 0.010, 0.754, 0.027),
    ("Chile", 2.040, 0.010, 1.520, 0.010, 0.503, 0.013),
    ("Czech Republic", 1.968, 0.008, 1.477, 0.006, 0.808, 0.011),
    ("Denmark", 2.205, 0.007, 1.504, 0.005, 0.861, 0.015),
    ("Finland", 2.222, 0.008, 1.595, 0.007, 0.985, 0.009),
    ("France", 2.050, 0.007, 1.534, 0.005, 0.915, 0.008),
    ("Germany", 1.903, 0.006, 1.431, 0.004, 0.717, 0.009),
    ("Greece", 0.331, 0.045, 1.460, 0.031, 0.903, 0.035),
    ("Hong Kong", 2.168, 0.009, 1.384, 0.007, 0.589, 0.020),
    ("Hungary", 0.925, 0.007, 1.008, 0.001, 0.534, 0.001),
    ("Iceland", -0.112, 0.075, 1.441, 0.031, 1.551, 0.044),
    ("Israel", 1.914, 0.005, 1.442, 0.003, 0.705, 0.008),
    ("Italy", 3.348, 0.004, 1.415, 0.006, 0.703, 0.012),
    ("Japan", 1.932, 0.004, 1.441, 0.003, 0.714, 0.009),
    ("Mexico", 1.644, 0.016, 1.563, 0.008, 0.988, 0.016),
    ("Norway", 1.953, 0.011, 1.468, 0.007, 0.779, 0.016),
    ("Pakistan", 1.198, 0.042, 1.501, 0.006, 0.946, 0.030),
    ("Poland", 2.317, 0.008, 1.184, 0.007, 0.669, 0.024),
    ("Slovenia", 1.828, 0.006, 1.383, 0.003, 0.349, 0.009),
    ("South Africa", 1.840, 0.013, 1.388, 0.008, 0.645, 0.018),
    ("Spain", 1.952, 0.011, 1.467, 0.007, 0.784, 0.015),
    ("Sweden", 2.032, 0.011, 1.520, 0.007, 0.893, 0.015),
    ("Switzerland", 1.957, 0.010, 1.471, 0.007, 0.779, 0.017),
]

# Preregistered 10-country subset: decile-stratified on lambda.
SELECTED = ["Iceland", "Greece", "Brazil", "Hungary", "Pakistan",
            "Australia", "Germany", "Japan", "Denmark", "Italy"]

N_MEDIAN = 150  # median INTRA per-country n (~7000/53)
SQRT_N = math.sqrt(N_MEDIAN)


def _winsor(x, lo, hi):
    return max(lo, min(hi, x))


class CountryDist:
    """Reconstructed within-country distribution of CPT parameters."""

    def __init__(self, row, sd_scale=1.0):
        self.name = row[0]
        self.lam = row[1]  # loss aversion
        self.alp = row[3]  # curvature
        self.dlt = row[5]  # probability weighting
        self.sd_lam = _winsor(row[2] * SQRT_N * sd_scale, 0.05, 0.50)
        self.sd_alp = _winsor(row[4] * SQRT_N * sd_scale, 0.03, 0.25)
        self.sd_dlt = _winsor(row[6] * SQRT_N * sd_scale, 0.05, 0.40)

    def draw(self, rng):
        lam = float(rng.normal(self.lam, self.sd_lam))
        lam = max(0.02, lam)  # truncate; disclosed (Iceland published < 0)
        alp = max(0.3, float(rng.normal(self.alp, self.sd_alp)))
        dlt = max(0.1, float(rng.normal(self.dlt, self.sd_dlt)))
        return lam, alp, dlt


DISTS = {r[0]: CountryDist(r) for r in RAW}


def draw_individual(rng, country=None):
    """Region: country given. Generic (country=None): pooled 27-country mixture."""
    if country is None:
        pick = RAW[rng.integers(len(RAW))][0]
        return DISTS[pick].draw(rng)
    return DISTS[country].draw(rng)


# ---- CPT prediction on held-out tasks ----
# Parameterization note: the INTRA papers' "risk aversion" estimate phi > 1
# denotes a CONCAVE value function (universal gain risk aversion, Rieger et al.
# 2015). Hence v(x) = x^(1/phi), i.e. exponent s = 1/phi < 1. Using x^phi
# directly would invert the published finding; verified against the paper's
# headline result (risk aversion in gains in all 53 countries).
def w(p, delta):
    return p ** delta / (p ** delta + (1 - p) ** delta) ** (1.0 / delta)


def v(x, lam, phi):
    s = 1.0 / phi
    return x ** s if x >= 0 else -lam * ((-x) ** s)


# ---- Elicitation v3 (amended 2026-10-08): Yes/No WTP tasks ----
# The binary A/B lottery format produced degenerate label bias under the chat
# template and content-insensitive choices under raw completion (diagnostics
# /tmp/lp*.out). Validated replacement: qualitative 3-level personas (semantic,
# not numerical) + Yes/No willingness questions. P(yes) is monotone decreasing
# in persona intensity for both gains and losses (lp10/lp11).
#
# (task_id, domain, price): gain = "Would you pay $price for a 50/50 chance to
# win $100?"; loss = "Would you accept a 50/50 chance of losing $100 if paid
# $price?". Prices fixed across individuals; span the human CE/WTA range.
TASKS = [
    ("G_25", "gain", 25),
    ("G_45", "gain", 45),
    ("G_65", "gain", 65),
    ("L_40", "loss", 40),
    ("L_90", "loss", 90),
    ("L_140", "loss", 140),
]


def cpt_yes(lam, phi, task):
    """Deterministic CPT-predicted Yes/No per individual.

    Gain: yes iff price <= CE, CE = 100 * 0.5^phi.
    Loss: yes iff price >= WTA, WTA = 100 * (0.5*lam)^phi.
    (Same CE/WTA formulas as the v1 profile_numbers; objective 0.5 disclosed.)
    """
    _, domain, price = task
    if domain == "gain":
        return price <= 100 * 0.5 ** phi
    wta = 100 * (0.5 * lam) ** phi
    return price >= wta


def ev_yes(task):
    """EV-optimal Yes/No: gain EV = $50, loss EV = -$50."""
    _, domain, price = task
    return price <= 50 if domain == "gain" else price >= 50


# ---- Persona-level mapping (v3) ----
# Absolute tertile cutoffs from the pooled 27-country reconstructed individual
# distribution (seed 0). Absolute (not within-country) so that country mixtures
# of levels differ -> between-country heterogeneity is preserved.
PHI_TERTILES = [1.394, 1.490]
LAM_TERTILES = [1.801, 2.007]


def risk_level(phi):
    """0 = least risk-averse ... 2 = most risk-averse (from curvature phi)."""
    if phi < PHI_TERTILES[0]:
        return 0
    if phi < PHI_TERTILES[1]:
        return 1
    return 2


def loss_level(lam):
    """0 = least loss-averse ... 2 = most loss-averse (from lambda)."""
    if lam < LAM_TERTILES[0]:
        return 0
    if lam < LAM_TERTILES[1]:
        return 1
    return 2


RISK_SENTENCES = [
    "You are comfortable with uncertainty. You often prefer a gamble over a sure amount.",
    "You are somewhat cautious with money. You usually prefer sure amounts but sometimes take gambles.",
    "You are very cautious with money. You strongly prefer guaranteed amounts over any gamble.",
]

LOSS_SENTENCES = [
    "You do not mind risking losses. You will take a gamble that could lose money if the compensation is reasonable.",
    "You dislike losing money. You need good compensation to accept a gamble where you could lose.",
    "You refuse to risk losing money. You would need an extremely large payment to accept any gamble where you could lose.",
]


def profile_numbers(lam, phi):
    """Behavioral profile numbers from a DRAWN (truncated, lam >= 0.02) individual.

    CE_gain solves ce^(1/phi) = 0.5 * 100^(1/phi)  ->  ce = 100 * 0.5^phi.
    WTA_loss solves wta^(1/phi) = 0.5 * lam * 100^(1/phi)  ->  wta = 100*(0.5*lam)^phi.
    Objective 0.5 used for the profile description (disclosed assumption).
    """
    ce_gain = int(round(100 * 0.5 ** phi))
    wta_loss = max(1, int(round(100 * (0.5 * lam) ** phi)))
    return ce_gain, wta_loss
