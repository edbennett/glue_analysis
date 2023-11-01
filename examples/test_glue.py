#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from glue_analysis.readers.read_fortran import read_correlators_fortran

correlators = read_correlators_fortran(
    "raw_data/nf2_adj/b2.35/m-1.09/96x48/out_corr_torelon",
    vev_filename="raw_data/nf2_adj/b2.35/m-1.09/96x48/out_vev_torelon",
)

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlabel("$t$")
ax.set_ylabel(r"$m_{\mathrm{eff}}$")
ax.set_title("Sorting by eigenvalue, vacuum subtracted")

for subtract in True, False:
    correlators_pe = correlators.get_pyerrors(subtract=subtract)
    correlators_pe.gamma_method()

    gs_correlator = correlators_pe.Eigenvalue(t0=0)

    m_eff = gs_correlator.m_eff(variant="log")
    m_eff.gamma_method()
    t, m, m_err = m_eff.plottable()
    ax.errorbar(
        [ti + 1 + (0.2 if subtract else 0) for ti in t],
        m,
        yerr=m_err,
        label=f"Vacuum {'not ' if not subtract else ''}subtracted",
        fmt="." if subtract else "x",
    )

ax.set_ylim(-0.05, 0.75)
ax.legend(loc="best")

fig.savefig("string_vacuumsubtracted.pdf")
plt.close(fig)
