#!/usr/bin/env python3
"""
plot_pdos_template.py -- pDOS da VASP, formato riusabile fra manoscritti.

Generalizzazione di plot_pdos.py (figure Gr/CuO e Gr/AgO). Stesse convenzioni
grafiche, niente piu' cucito sul sistema: nessun elemento nel codice, nessun
conteggio di atomi fisso, nessuna modalita' legata a un'affermazione specifica.

CONVENZIONI, che sono la parte da riusare
  * Energie riferite a E_F (o al VBM con --ref vbm), riga verticale a zero.
  * Spin su positivo, spin giu' negativo e specchiato, con la riga orizzontale
    a zero: la simmetria di un sottoreticolo AFM si vede a occhio.
  * Curve riempite con bordo piu' scuro; un solo asse, nessun asse doppio.
  * Palette sicura per daltonismo; ogni serie ha anche uno stile di linea
    proprio, cosi' la figura sopravvive alla stampa in grigio.
  * ISPIN = 1 non ha il canale giu': l'asse diventa a un lato solo invece di
    lasciare mezzo pannello vuoto con l'etichetta "spin giu'" sul nulla.
  * Gruppi minuscoli (un adsorbato contro uno slab) vengono ingranditi di un
    fattore scritto NELLA LEGENDA: senza il fattore in legenda la figura e'
    scorretta, con il fattore e' una scelta di leggibilita'.

IL CENTRO DI BANDA VUOLE LA SUA FINESTRA, SEMPRE SCRITTA
  epsilon = integral E rho(E) dE / integral rho(E) dE, sulla finestra data.
  Il valore DIPENDE dalla finestra, e non poco: nel manoscritto AJIF lo stesso
  centro d usciva -3.49 eV su [-6, 4] e -3.63 eV su [-8, 4], e il confronto
  fra due sistemi cambiava SEGNO fra le due scelte. Quindi qui la finestra e'
  un argomento obbligato nel senso che compare sempre: nell'annotazione, nello
  stdout e nel CSV. Due sistemi confrontati nello stesso testo vanno calcolati
  sulla stessa finestra, e la finestra va dichiarata nella didascalia.
  Il default [-inf, 0] e' il centro della banda OCCUPATA, che e' la definizione
  usata da Hammer-Norskov; --centre-window -8 4 da' la banda intera.

COSA LEGGE
  vasprun.xml  preferito: porta E_F, proiezioni orbitali e canali di spin in un
               solo file. Serve LORBIT = 11 nel calcolo che l'ha prodotto.
  DOSCAR+POSCAR  fallback.
  OUTCAR       opzionale, per i momenti per sito: servono a separare i
               sottoreticoli di un antiferromagnete (vedi --sublattice).
  Il parser XML e' scritto qui dentro: pymatgen si usa se c'e', ma NON serve.
  Nessuna dipendenza oltre numpy e matplotlib.

UNA NOTA SUI DATI, NON SULLO SCRIPT
  Una DOS liscia vuole una maglia k piu' fitta di quella di un rilassamento.
  Se le curve sono spigolose il problema non e' lo smoothing: rifai il calcolo
  non autoconsistente (ICHARG = 11 su un CHGCAR convergito, NEDOS = 3001,
  KPOINTS piu' fitti). Lo smoothing gaussiano di --smooth e' cosmetico e non
  sostituisce la maglia: allarga i picchi e abbassa i massimi.

USO
  # un sistema, decomposizione per elemento e orbitale
  python3 plot_pdos_template.py scf_dir \\
      --group "Cu 3d=Cu:d" --group "O 2p=O:p" --group "C 2p=C:p" \\
      --centre "Cu 3d" --centre-window -8 4 --out fig_pdos

  # piu' sistemi affiancati, stessa scala
  python3 plot_pdos_template.py dirA dirB dirC \\
      --group "metal d=Cu:d" --group "O 2p=O:p" --share-y --out fig_compare

  # antiferromagnete: separa i due sottoreticoli dai momenti in OUTCAR
  python3 plot_pdos_template.py afm_dir --sublattice Cu:d \\
      --group "O 2p=O:p" --out fig_afm

  # adsorbato per indice (1-based, come in VESTA), ingrandito
  python3 plot_pdos_template.py ads_dir \\
      --group "slab Cu d=Cu:d" --group "ads C 2p=#63:p" \\
      --group "ads O 2p=#64-65:p" --magnify "ads C 2p" --magnify "ads O 2p"
"""
import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator
except ImportError:
    sys.exit("servono numpy e matplotlib")


# =========================================================== stile
PALETTE = ["#1f4e79", "#c1121f", "#2a9d8f", "#e9c46a", "#7209b7",
           "#4a4a4a", "#f4a261", "#457b9d"]
LINESTYLES = ["-", "-", "-", "--", "--", ":", ":", "-."]
FERMI_C = "#000000"
GAP_C = "#f2f2f2"


def set_style(base=9):
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": base,
        "axes.labelsize": base + 1, "axes.titlesize": base + 1.5,
        "axes.titleweight": "bold", "axes.linewidth": 0.9,
        "xtick.labelsize": base - 0.5, "ytick.labelsize": base - 0.5,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "legend.fontsize": base - 1, "legend.frameon": False,
        "mathtext.default": "regular",
    })


# =========================================================== lettura
# Nomi dei campi che VASP scrive in <partial>, raggruppati in s/p/d/f.
ORB_OF = {}
for _o in ("s",):
    ORB_OF[_o] = "s"
for _o in ("py", "pz", "px", "p"):
    ORB_OF[_o] = "p"
for _o in ("dxy", "dyz", "dz2", "dxz", "dx2", "dx2-y2", "d"):
    ORB_OF[_o] = "d"
for _o in ("f-3", "f-2", "f-1", "f0", "f1", "f2", "f3", "f"):
    ORB_OF[_o] = "f"


def _blank(E):
    return {"up": np.zeros_like(E), "down": np.zeros_like(E)}


def read_vasprun_xml(path):
    """Parser diretto di vasprun.xml: energie, E_F, simboli, proiezioni.

    Scritto a mano invece di affidarsi a pymatgen perche' un template deve
    girare su una macchina dove pymatgen non c'e'. La struttura e':
      <atominfo><array name="atoms"><set><rc><c>Cu</c>...
      <calculation><dos><i name="efermi">
        <partial><array><field>energy</field><field>s</field>...
          <set><set comment="ion 1"><set comment="spin 1"><r> ... </r>
    """
    try:
        tree = ET.parse(path)
    except (ET.ParseError, OSError) as e:
        print(f"  [vasprun.xml illeggibile: {e}]")
        return None
    root = tree.getroot()

    symbols = []
    for arr in root.iter("array"):
        if arr.get("name") == "atoms":
            for rc in arr.iter("rc"):
                cs = rc.findall("c")
                if cs and cs[0].text:
                    symbols.append(cs[0].text.strip())
            break
    if not symbols:
        print("  [vasprun.xml: nessun elenco di atomi]")
        return None

    # l'ultima <dos> e' quella del calcolo finale
    dos = None
    for d in root.iter("dos"):
        dos = d
    if dos is None:
        print("  [vasprun.xml: nessun blocco <dos>]")
        return None

    efermi = None
    for i in dos.findall("i"):
        if i.get("name") == "efermi" and i.text:
            try:
                efermi = float(i.text)
            except ValueError:
                pass
    if efermi is None:
        print("  [vasprun.xml: nessun efermi]")
        return None

    # --- DOS totale, per il gap e per il controllo di spin ---
    E = None
    tot_up = tot_dn = None
    total = dos.find("total")
    if total is not None:
        sets = [s for s in total.iter("set") if s.get("comment", "").startswith("spin")]
        for k, s in enumerate(sets):
            rows = np.array([[float(x) for x in r.text.split()]
                             for r in s.findall("r") if r.text])
            if rows.size == 0:
                continue
            if E is None:
                E = rows[:, 0]
            if k == 0:
                tot_up = rows[:, 1]
            elif k == 1:
                tot_dn = rows[:, 1]
    if E is None:
        print("  [vasprun.xml: DOS totale vuota]")
        return None
    if tot_dn is None:
        tot_dn = np.zeros_like(E)

    # --- proiezioni per ione ---
    partial = dos.find("partial")
    proj = [{} for _ in symbols]
    spin_pol = tot_dn.any()
    if partial is None:
        print("  [vasprun.xml senza <partial>: serve LORBIT = 11. "
              "Solo DOS totale disponibile]")
    else:
        fields = [f.text.strip() for f in partial.iter("field") if f.text]
        # il primo campo e' 'energy'
        orbs = fields[1:] if fields and fields[0].lower().startswith("ener") else fields
        ion_sets = [s for s in partial.iter("set")
                    if s.get("comment", "").startswith("ion")]
        for s in ion_sets:
            m = re.search(r"ion\s+(\d+)", s.get("comment", ""))
            if not m:
                continue
            ia = int(m.group(1)) - 1
            if ia >= len(symbols):
                continue
            entry = {}
            spin_sets = [ss for ss in s.iter("set")
                         if ss.get("comment", "").startswith("spin")]
            for k, ss in enumerate(spin_sets):
                rows = np.array([[float(x) for x in r.text.split()]
                                 for r in ss.findall("r") if r.text])
                if rows.size == 0:
                    continue
                chan = "up" if k == 0 else "down"
                if k > 1:
                    continue
                for j, oname in enumerate(orbs):
                    grp = ORB_OF.get(oname)
                    if grp is None or j + 1 >= rows.shape[1]:
                        continue
                    entry.setdefault(grp, _blank(E))
                    entry[grp][chan] = entry[grp][chan] + rows[:, j + 1]
            proj[ia] = entry
            if len(spin_sets) > 1:
                spin_pol = True

    return {"E": E, "efermi": efermi, "symbols": symbols, "proj": proj,
            "total": tot_up + tot_dn if tot_up is not None else None,
            "spin_polarised": bool(spin_pol)}


def read_vasprun_pymatgen(path):
    """Se pymatgen c'e', si usa: gestisce varianti di VASP che il parser a mano
    non conosce. Non e' un requisito."""
    try:
        from pymatgen.io.vasp.outputs import Vasprun
        from pymatgen.electronic_structure.core import Spin, OrbitalType
    except ImportError:
        return None
    try:
        vr = Vasprun(path, parse_potcar_file=False)
        cdos = vr.complete_dos
    except Exception as e:
        print(f"  [pymatgen ha fallito: {e}]")
        return None
    E = np.asarray(cdos.energies)
    proj = []
    for site in cdos.structure:
        sd = cdos.get_site_spd_dos(site)
        entry = {}
        for ot, key in ((OrbitalType.s, "s"), (OrbitalType.p, "p"),
                        (OrbitalType.d, "d")):
            if ot in sd:
                d = sd[ot].densities
                entry[key] = {"up": np.asarray(d.get(Spin.up, np.zeros_like(E))),
                              "down": np.asarray(d.get(Spin.down,
                                                       np.zeros_like(E)))}
        proj.append(entry)
    tot = cdos.densities
    total = sum(np.asarray(v) for v in tot.values())
    return {"E": E, "efermi": float(cdos.efermi),
            "symbols": [str(s.specie) for s in cdos.structure],
            "proj": proj, "total": total, "spin_polarised": len(tot) > 1}


def read_doscar(directory):
    """Fallback DOSCAR + POSCAR. Le colonne di DOSCAR dipendono da LORBIT e da
    ISPIN e non sono autodescrittive: per questo vasprun.xml e' preferito."""
    dpath = os.path.join(directory, "DOSCAR")
    ppath = next((os.path.join(directory, n) for n in ("POSCAR", "CONTCAR")
                  if os.path.isfile(os.path.join(directory, n))), None)
    if not (os.path.isfile(dpath) and ppath):
        return None
    pl = open(ppath, errors="ignore").read().splitlines()
    els, cnt = pl[5].split(), [int(x) for x in pl[6].split()]
    symbols = [e for e, c in zip(els, cnt) for _ in range(c)]

    lines = open(dpath, errors="ignore").read().splitlines()
    natoms = int(lines[0].split()[0])
    hdr = lines[5].split()
    nedos, efermi = int(hdr[2]), float(hdr[3])
    body = [l.split() for l in lines[6:]]
    spin = len(body[0]) >= 5
    E = np.array([float(r[0]) for r in body[:nedos]])
    total = np.array([float(r[1]) + (float(r[2]) if spin else 0.0)
                      for r in body[:nedos]])

    proj, idx = [], nedos
    for _ in range(natoms):
        idx += 1
        blk = body[idx:idx + nedos]
        idx += nedos
        if len(blk) < nedos:
            proj.append({})
            continue
        b = np.array([[float(x) for x in r] for r in blk])
        ncol = b.shape[1] - 1
        entry = {}
        if spin and ncol >= 18:
            entry = {
                "s": {"up": b[:, 1], "down": b[:, 2]},
                "p": {"up": b[:, 3::2][:, :3].sum(1),
                      "down": b[:, 4::2][:, :3].sum(1)},
                "d": {"up": b[:, 9::2][:, :5].sum(1),
                      "down": b[:, 10::2][:, :5].sum(1)}}
        elif not spin and ncol >= 9:
            z = np.zeros(nedos)
            entry = {"s": {"up": b[:, 1], "down": z},
                     "p": {"up": b[:, 2:5].sum(1), "down": z},
                     "d": {"up": b[:, 5:10].sum(1), "down": z}}
        proj.append(entry)
    return {"E": E, "efermi": efermi, "symbols": symbols, "proj": proj,
            "total": total, "spin_polarised": spin}


def site_moments(directory):
    """Momenti totali per ione dall'ultima tabella 'magnetization (x)'.

    Indispensabili per un antiferromagnete: sommata su TUTTI i siti del
    metallo l'asimmetria di spin di un AFM si annulla esattamente, quindi il
    grafico sarebbe identico a quello di un non magnete. Separati per
    sottoreticolo, i due canali speculari sono la firma dello stato AFM.
    """
    p = os.path.join(directory, "OUTCAR")
    if not os.path.isfile(p):
        return None
    tables, cur, inm = [], [], False
    for line in open(p, errors="ignore"):
        if "magnetization (x)" in line:
            inm, cur = True, []
            continue
        if inm:
            t = line.split()
            if len(t) == 5 and t[0].isdigit():
                cur.append(float(t[-1]))
            elif cur and ("---" in line or line.strip().startswith("tot")):
                tables.append(cur)
                inm = False
    return tables[-1] if tables else None


def load(directory, ref="fermi", prefer_pymatgen=False):
    vr = os.path.join(directory, "vasprun.xml")
    d = None
    if os.path.isfile(vr):
        if prefer_pymatgen:
            d = read_vasprun_pymatgen(vr)
        if d is None:
            d = read_vasprun_xml(vr)
        if d is None and not prefer_pymatgen:
            d = read_vasprun_pymatgen(vr)
    if d is None:
        d = read_doscar(directory)
    if d is None:
        raise FileNotFoundError(
            f"ne' vasprun.xml ne' DOSCAR+POSCAR leggibili in {directory}")

    zero = d["efermi"]
    if ref == "vbm":
        # Per un semiconduttore E_F scritto da VASP cade dentro il gap in un
        # punto che dipende dallo smearing: il VBM e' un riferimento piu'
        # stabile fra calcoli diversi.
        tot = d.get("total")
        if tot is not None:
            occ = d["E"][(d["E"] <= d["efermi"]) & (tot > 1e-3)]
            if occ.size:
                zero = float(occ.max())
    d["E"] = d["E"] - zero
    d["zero_abs"] = zero
    d["ref"] = ref
    return d


# =========================================================== selezione
def parse_group(spec, symbols):
    """'LABEL=SEL[,SEL...]' -> (label, set di indici, orbitale)

    SEL e' 'El:orb' oppure '#i:orb' / '#i-j:orb' con indici 1-based, cioe'
    gli stessi numeri che mostrano VESTA e il POSCAR. Mescolare gli stili
    1-based e 0-based fra script e visualizzatore e' una delle fonti di
    errore piu' banali e piu' difficili da vedere in una figura.
    """
    if "=" not in spec:
        raise ValueError(f"gruppo senza '=': {spec}")
    label, sels = spec.split("=", 1)
    idx, orb = set(), None
    for sel in sels.split(","):
        sel = sel.strip()
        if ":" not in sel:
            raise ValueError(f"selettore senza ':orbitale': {sel}")
        who, o = sel.rsplit(":", 1)
        o = o.strip()
        if o not in ("s", "p", "d", "f"):
            raise ValueError(f"orbitale non riconosciuto: {o}")
        if orb is not None and orb != o:
            raise ValueError(f"gruppo '{label}' mescola orbitali diversi")
        orb = o
        who = who.strip()
        if who.startswith("#"):
            rng = who[1:]
            if "-" in rng:
                a, b = rng.split("-", 1)
                idx |= {i - 1 for i in range(int(a), int(b) + 1)}
            else:
                idx.add(int(rng) - 1)
        else:
            idx |= {i for i, s in enumerate(symbols) if s == who}
    bad = {i for i in idx if i < 0 or i >= len(symbols)}
    if bad:
        raise ValueError(f"gruppo '{label}': indici fuori cella {sorted(bad)}")
    if not idx:
        raise ValueError(f"gruppo '{label}': nessun atomo selezionato")
    return label.strip(), idx, orb


def sum_proj(data, indices, orbital):
    E = data["E"]
    up = np.zeros_like(E)
    dn = np.zeros_like(E)
    for i in indices:
        e = data["proj"][i] if i < len(data["proj"]) else {}
        if orbital in e:
            up = up + e[orbital]["up"]
            dn = dn + e[orbital]["down"]
    return up, dn


# =========================================================== analisi
def _trap(y, x):
    f = getattr(np, "trapezoid", None) or np.trapz
    return f(y, x)


def band_centre(E, dos, lo=None, hi=0.0):
    """Primo momento della DOS sulla finestra [lo, hi], in eV rispetto allo zero.

    Restituisce anche il peso integrato: un centro calcolato su un peso
    trascurabile e' un numero senza significato, e senza il peso non si vede.
    """
    m = np.ones_like(E, dtype=bool)
    if lo is not None:
        m &= E >= lo
    if hi is not None:
        m &= E <= hi
    if not m.any():
        return np.nan, 0.0
    w = _trap(dos[m], E[m])
    if abs(w) < 1e-12:
        return np.nan, float(w)
    return float(_trap(E[m] * dos[m], E[m]) / w), float(w)


def band_gap(E, total, tol=1e-3):
    below = E[(E < 0) & (total > tol)]
    above = E[(E > 0) & (total > tol)]
    if below.size == 0 or above.size == 0:
        return np.nan
    return float(above.min() - below.max())


def smooth(y, sigma_pts=3):
    """Convoluzione gaussiana. Cosmetica: allarga i picchi e abbassa i massimi,
    quindi non va usata per leggere altezze dal grafico."""
    if sigma_pts <= 0:
        return y
    x = np.arange(-3 * sigma_pts, 3 * sigma_pts + 1)
    k = np.exp(-0.5 * (x / sigma_pts) ** 2)
    k /= k.sum()
    return np.convolve(y, k, mode="same")


# =========================================================== disegno
def pick_legend_corner(E, series, xlim):
    """Angolo in alto con meno curve sotto.

    La riga di E_F e la sua etichetta stanno sempre a x = 0, che in una
    finestra tipo [-8, 4] cade nel terzo destro: una legenda fissa in alto a
    destra ci finisce sopra. Si guarda il peso nei due quarti esterni e si
    sceglie il lato piu' vuoto.
    """
    lo, hi = xlim
    q = (hi - lo) / 4.0
    left = (E >= lo) & (E <= lo + q)
    right = (E >= hi - q) & (E <= hi)
    wl = wr = 0.0
    for s in series:
        tot = np.abs(s["up"]) + np.abs(s["down"])
        if left.any():
            wl += float(tot[left].sum())
        if right.any():
            wr += float(tot[right].sum())
    return "upper left" if wl <= wr else "upper right"


def draw_panel(ax, E, series, xlim, smooth_pts=3, gap=None, centre=None,
               title=None, legend_loc="auto", ylabel=True, ymax=None,
               force_symmetric=False):
    if legend_loc == "auto":
        legend_loc = pick_legend_corner(E, series, xlim)
    peak = 0.0
    for s in series:
        up = smooth(s["up"], smooth_pts)
        dn = -smooth(s["down"], smooth_pts)
        ax.fill_between(E, 0, up, color=s["colour"], alpha=s.get("alpha", 0.5),
                        linewidth=0, zorder=2)
        ax.plot(E, up, color=s["colour"], lw=1.1, ls=s.get("ls", "-"),
                label=s["label"], zorder=3)
        if np.any(np.abs(dn) > 1e-9):
            ax.fill_between(E, 0, dn, color=s["colour"],
                            alpha=s.get("alpha", 0.5), linewidth=0, zorder=2)
            ax.plot(E, dn, color=s["colour"], lw=1.1, ls=s.get("ls", "-"),
                    zorder=3)
        peak = max(peak, np.abs(up).max(), np.abs(dn).max())
    peak = ymax or peak or 1.0

    if gap is not None and np.isfinite(gap) and gap > 0.05:
        ax.axvspan(-gap / 2, gap / 2, color=GAP_C, zorder=0)
        ax.annotate(f"gap $\\approx$ {gap:.2f} eV", xy=(0, -1.02 * peak),
                    ha="center", va="top", fontsize=7.5, color="0.30")
    ax.axhline(0, color="0.25", lw=0.8, zorder=1)
    ax.axvline(0, color=FERMI_C, lw=1.2, ls="--", zorder=4)
    ax.annotate("$E_\\mathrm{F}$", xy=(0, 0), xytext=(4, 4),
                textcoords="offset points", fontsize=8, color=FERMI_C,
                ha="left", va="bottom", zorder=5)

    if centre is not None:
        val, lo, hi, lab, col = centre
        if np.isfinite(val):
            ax.axvline(val, color=col, lw=1.0, ls=":", zorder=4)
            win = (f"[{lo:g}, {hi:g}]" if lo is not None else f"[occ, {hi:g}]")
            ax.annotate(f"$\\varepsilon$({lab}) = {val:.2f} eV\n{win} eV",
                        xy=(val, 0.99 * peak), xytext=(3, -2),
                        textcoords="offset points", ha="left", va="top",
                        fontsize=7.5, color=col,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="none", alpha=0.75))

    ax.set_xlim(*xlim)
    # Un pannello a un lato e uno a due lati, alti uguali, disegnano la stessa
    # densita' a scale diverse: confrontarli a occhio e' fuorviante. Con
    # force_symmetric tutti i pannelli usano l'asse a due lati.
    has_down = any(np.any(np.abs(s["down"]) > 1e-9) for s in series)
    two_sided = has_down or force_symmetric
    ax.set_ylim(-1.15 * peak, 1.15 * peak) if two_sided else ax.set_ylim(0, 1.15 * peak)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    if ylabel:
        ax.set_ylabel("pDOS (stati eV$^{-1}$)")
    if title:
        ax.set_title(title, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    sx, sha = (0.015, "left") if "right" in legend_loc else (0.985, "right")
    if has_down:
        ax.text(sx, 0.93, "spin $\\uparrow$", transform=ax.transAxes,
                fontsize=7.5, ha=sha, color="0.35")
        ax.text(sx, 0.05, "spin $\\downarrow$", transform=ax.transAxes,
                fontsize=7.5, ha=sha, color="0.35")
    else:
        ax.text(sx, 0.93, "spin-restricted", transform=ax.transAxes,
                fontsize=7.5, ha=sha, color="0.35")
    ax.legend(loc=legend_loc, ncol=1)
    return peak


def save(fig, stem, formats=("pdf", "png")):
    fig.tight_layout()
    for ext in formats:
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  scritto {stem}." + "/.".join(formats))


# =========================================================== main
def build_series(d, args, directory):
    """Serie da disegnare per una directory, piu' le righe per il CSV."""
    series, csv_rows, centre = [], [], None
    ci = 0

    if args.sublattice:
        who, orb = args.sublattice.rsplit(":", 1)
        midx = [i for i, s in enumerate(d["symbols"]) if s == who]
        mom = site_moments(directory)
        if mom and len(mom) > (max(midx) if midx else -1):
            A = [i for i in midx if mom[i] > args.moment_tol]
            B = [i for i in midx if mom[i] < -args.moment_tol]
            if A and B:
                for name, sel, st in ((f"{who} {orb}, sublat. A ($m>0$, n={len(A)})", A, "-"),
                                      (f"{who} {orb}, sublat. B ($m<0$, n={len(B)})", B, "--")):
                    u, v = sum_proj(d, set(sel), orb)
                    series.append({"label": name, "up": u, "down": v,
                                   "colour": PALETTE[ci % len(PALETTE)],
                                   "ls": st, "alpha": 0.5})
                    csv_rows.append((name, u, v))
                    ci += 1
                print(f"    sottoreticoli: {len(A)} su / {len(B)} giu'")
            else:
                print(f"    momenti tutti dello stesso segno (tol "
                      f"{args.moment_tol}): nessuna separazione AFM")
        else:
            print("    ATTENZIONE: momenti per sito non disponibili in OUTCAR. "
                  "Sommata su tutti i siti, l'asimmetria di spin di un AFM si "
                  "annulla: il pannello non distinguera' AFM da non magnetico.")

    groups = []
    for spec in args.group:
        groups.append(parse_group(spec, d["symbols"]))

    # Un gruppo che somma a zero NON va disegnato come curva piatta: una linea
    # a zero si legge come "nessuno stato a questa energia", che e'
    # un'affermazione fisica, mentre qui vuol dire solo che la proiezione non
    # c'e' nel file (LORBIT sbagliato, orbitale assente per quell'elemento,
    # indici sbagliati). Meglio fermarsi che pubblicare una figura muta.
    raw = {}
    empty = []
    for label, idx, orb in groups:
        u, v = sum_proj(d, idx, orb)
        if not np.any(np.abs(u) > 1e-12) and not np.any(np.abs(v) > 1e-12):
            empty.append((label, orb, sorted(idx)[:6]))
        raw[label] = (u, v, len(idx))
    if empty:
        for label, orb, sample in empty:
            print(f"    VUOTO: '{label}' (orbitale {orb}, atomi "
                  f"{[i + 1 for i in sample]}...) non ha proiezione nel file")
        if len(empty) == len(groups):
            raise ValueError(
                "tutti i gruppi sono vuoti: il file non ha le proiezioni "
                "richieste. Controlla LORBIT = 11 nel calcolo, e che gli "
                "elementi e gli indici (1-based) corrispondano al POSCAR")

    # ingrandimento dei gruppi piccoli, con il fattore scritto in legenda
    ref_peak = max((np.max(u + v) for u, v, _ in raw.values()), default=1.0)
    for label, idx, orb in groups:
        u, v, n = raw[label]
        sc = 1.0
        if label in args.magnify:
            pk = max(np.max(u + v), 1e-12)
            cand = args.magnify_target * ref_peak / pk
            sc = 1.0 if cand <= 1.5 else float(min(args.magnify_max, round(cand)))
        lab = label if sc == 1.0 else f"{label} ($\\times${sc:.0f})"
        series.append({"label": lab, "up": u * sc, "down": v * sc,
                       "colour": PALETTE[ci % len(PALETTE)],
                       "ls": LINESTYLES[ci % len(LINESTYLES)], "alpha": 0.5})
        csv_rows.append((label, u, v))     # nel CSV i valori NON scalati
        if args.centre and label == args.centre:
            val, w = band_centre(d["E"], u + v, args.centre_window[0],
                                 args.centre_window[1])
            centre = (val, args.centre_window[0], args.centre_window[1],
                      label, PALETTE[ci % len(PALETTE)])
            print(f"    centro({label}) = {val:.4f} eV   finestra "
                  f"[{args.centre_window[0]}, {args.centre_window[1]}] eV   "
                  f"peso {w:.3f} stati")
            if abs(w) < 0.1:
                print("      peso trascurabile: il centro non e' significativo")
        ci += 1
    return series, csv_rows, centre


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__.split("USO")[0])
    ap.add_argument("dirs", nargs="+", help="directory con vasprun.xml o DOSCAR")
    ap.add_argument("--group", action="append", default=[], metavar="LABEL=SEL",
                    help="serie da disegnare, es. \"Cu 3d=Cu:d\" oppure "
                         "\"ads O=#64-65:p\" (indici 1-based). Ripetibile")
    ap.add_argument("--sublattice", default=None, metavar="EL:ORB",
                    help="separa i sottoreticoli magnetici dai momenti in "
                         "OUTCAR, es. Cu:d")
    ap.add_argument("--moment-tol", type=float, default=0.05,
                    help="soglia in mu_B per assegnare un sito a un sottoreticolo")
    ap.add_argument("--centre", default=None, metavar="LABEL",
                    help="gruppo di cui calcolare il centro di banda")
    ap.add_argument("--centre-window", nargs=2, type=float,
                    default=[None, 0.0], metavar=("LO", "HI"),
                    help="finestra del centro in eV rispetto allo zero "
                         "(default: stati occupati). IL VALORE DIPENDE DALLA "
                         "FINESTRA: usa la stessa per tutti i sistemi che "
                         "confronti, e dichiarala nella didascalia")
    ap.add_argument("--ref", choices=["fermi", "vbm"], default="fermi",
                    help="riferimento dello zero")
    ap.add_argument("--xlim", nargs=2, type=float, default=[-8.0, 4.0])
    ap.add_argument("--smooth", type=int, default=3,
                    help="sigma gaussiana in punti di griglia (0 = niente)")
    ap.add_argument("--gap", action="store_true",
                    help="stima e disegna il gap dalla DOS totale")
    ap.add_argument("--magnify", action="append", default=[], metavar="LABEL",
                    help="ingrandisci questo gruppo; il fattore finisce in "
                         "legenda. Ripetibile")
    ap.add_argument("--magnify-target", type=float, default=0.45,
                    help="frazione del picco di riferimento da raggiungere")
    ap.add_argument("--magnify-max", type=float, default=50)
    ap.add_argument("--share-y", action="store_true",
                    help="stessa scala y su tutti i pannelli")
    ap.add_argument("--panel-width", type=float, default=4.8)
    ap.add_argument("--panel-height", type=float, default=3.6)
    ap.add_argument("--legend-loc", default="auto",
                    help="'auto' sceglie l'angolo piu' vuoto, "
                         "cosi' la legenda non finisce sopra la riga di E_F")
    ap.add_argument("--titles", nargs="*", default=None,
                    help="titoli dei pannelli; default i nomi delle directory")
    ap.add_argument("--suptitle", default=None)
    ap.add_argument("--font", type=float, default=9)
    ap.add_argument("--formats", nargs="*", default=["pdf", "png"])
    ap.add_argument("--csv", default=None,
                    help="scrive le curve non scalate, per rifare la figura "
                         "senza riparsare vasprun.xml")
    ap.add_argument("--prefer-pymatgen", action="store_true")
    ap.add_argument("--out", default="fig_pdos")
    args = ap.parse_args()

    if not args.group and not args.sublattice:
        sys.exit("serve almeno un --group oppure --sublattice")
    set_style(args.font)

    loaded, all_series, all_centres, all_csv = [], [], [], []
    for dd in args.dirs:
        print(f"{dd}")
        d = load(dd, args.ref, args.prefer_pymatgen)
        print(f"    {len(d['symbols'])} atomi, zero a {d['zero_abs']:.4f} eV "
              f"({d['ref']}), spin "
              f"{'polarizzato' if d['spin_polarised'] else 'ristretto'}")
        try:
            s, rows, c = build_series(d, args, dd)
        except ValueError as e:
            sys.exit(f"  {dd}: {e}")
        loaded.append(d)
        all_series.append(s)
        all_centres.append(c)
        all_csv.append(rows)

    ymax = None
    force_sym = False
    if args.share_y:
        ymax = max(max(np.abs(x["up"]).max(), np.abs(x["down"]).max())
                   for s in all_series for x in s)
        force_sym = any(np.any(np.abs(x["down"]) > 1e-9)
                        for s in all_series for x in s)

    n = len(args.dirs)
    fig, axes = plt.subplots(1, n, sharex=True,
                             figsize=(args.panel_width * n, args.panel_height))
    axes = np.atleast_1d(axes)
    titles = args.titles or [os.path.basename(os.path.normpath(x))
                             for x in args.dirs]
    for k, (ax, d, s, c) in enumerate(zip(axes, loaded, all_series, all_centres)):
        gap = band_gap(d["E"], smooth(d["total"], args.smooth), 0.05) \
            if (args.gap and d.get("total") is not None) else None
        draw_panel(ax, d["E"], s, tuple(args.xlim), args.smooth, gap, c,
                   titles[k] if k < len(titles) else None,
                   args.legend_loc, ylabel=(k == 0), ymax=ymax,
                   force_symmetric=force_sym)
        ax.set_xlabel("$E - E_\\mathrm{F}$ (eV)" if args.ref == "fermi"
                      else "$E - E_\\mathrm{VBM}$ (eV)")
    if args.suptitle:
        fig.suptitle(args.suptitle, fontsize=args.font + 1.5,
                     fontweight="bold", y=1.02)
    save(fig, args.out, tuple(args.formats))

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["directory", "group", "spin", "E_minus_ref_eV",
                        "pdos_states_per_eV"])
            for dd, d, rows in zip(args.dirs, loaded, all_csv):
                for label, u, v in rows:
                    for chan, arr in (("up", u), ("down", v)):
                        if chan == "down" and not np.any(np.abs(v) > 1e-9):
                            continue
                        for e, y in zip(d["E"], arr):
                            w.writerow([dd, label, chan, f"{e:.4f}", f"{y:.6g}"])
        print(f"  scritto {args.csv}  (valori NON scalati)")

    print("\nPer la didascalia: riferimento "
          + ("E_F" if args.ref == "fermi" else "VBM")
          + (f", centro di banda sulla finestra [{args.centre_window[0]}, "
             f"{args.centre_window[1]}] eV" if args.centre else "")
          + (f", smoothing gaussiano sigma = {args.smooth} punti"
             if args.smooth else ", nessuno smoothing"))


if __name__ == "__main__":
    main()
