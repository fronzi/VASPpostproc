#!/usr/bin/env python3
"""Kohn-Sham level-scheme template (vector PDF) -- the format of Figs. 5, 16, 17 of the NX0/NV- diamond paper.

One panel = one electronic configuration:  valence-band (VB) and conduction-band (CB) continua as soft gradients,
one column per spin channel, level bars with spin arrows, the energy (eV, relative to the VBM) printed under every
level, orbital labels at the left of each group, optional  (i) dashed excitation arrow, (ii) dimension bracket between
two levels, (iii) term-symbol / title in the corner.  Panels are laid out in a grid (rows = states, columns = methods,
defects, ...).  Everything is drawn from a plain dictionary, so adapting it to another paper means editing the data only.

Panel specification (see EXAMPLE at the bottom):
    spec = dict(
        vbm=0.0, cbm=4.20,                 # band edges in eV (same energy zero as the levels); zero is moved to the VBM
        zero=0.0,                          # value subtracted from every energy (e.g. the host-like level of the plot)
        term=r'$^{3}A_{2}$',               # optional text in the upper-left corner
        channels=[                         # one dict per spin channel, left to right
            dict(name='spin-up', color=BLUE, direction='up', levels=[
                    dict(label=r'$a_1$', e=0.83, occ=1),                   # non-degenerate level
                    dict(label=r'$e$',   e=(2.02, 2.02), occ=1)]),         # tuple -> a group of bars (e.g. an e pair)
            dict(name='spin-down', color=RED, direction='down', levels=[
                    dict(label=r'$a_1$', e=1.38, occ=1),
                    dict(label=r'$e$',   e=(3.26, 3.26), occ=0)])],
        excitation=((1, 0, 0), (1, 1, 0)),   # optional: ((channel, level, sub), (channel, level, sub)), dashed arrow
        bracket=dict(a=(1, 0, 0), b=(1, 1, 0), text=r'$\\Delta E_{a_1e}$ = 1.89 eV'),   # optional dimension line
    )
    occ may be a number (applied to every bar of the group) or a tuple with one entry per bar:
        1   -> one filled arrow (colour/direction of the channel)      0   -> empty bar
        0.5 -> fractional occupation: a '1/2' marker on the bar; an arrow is drawn on the lowest bar of the 0.5 bars.
Usage:  python3 level_scheme_template.py          (writes level_scheme_example.pdf)
Requires: numpy, matplotlib.  Font: Carlito if installed (setup(font=...) to change), otherwise DejaVu Sans.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import offset_copy
import matplotlib.patheffects as pe

# ----------------------------------------------------------------------------------------------- look
BLUE, RED, INK = '#1F5FA8', '#C8372D', '#1D2433'      # spin-up, spin-down, text/level colour (colour-blind safe pair)
MUTED, FAINT, HAIR = '#5E6A7D', '#8A94A6', '#D3DAE6'
BAND_EDGE, BAND_FAR, BAND_LINE, BAND_TXT = '#C3D1E8', '#E9EFF8', '#8EA6CF', '#4A5F8A'
HALO = [pe.withStroke(linewidth=2.2, foreground='white', alpha=0.85)]
GEOM = dict(bar=0.80, sub_gap=0.20, group_gap=0.50, left=0.55, right=0.35)   # data units of the x axis (level bars)
YLO = -1.4                                                                    # bottom of the panels (eV)


def setup(fontsize=8.5, font='Carlito'):
    """Matplotlib defaults.  `font` is used (also for the maths) if installed, otherwise DejaVu Sans."""
    from matplotlib import font_manager as fm
    try:
        fm.findfont(font, fallback_to_default=False)
        fam = [font, 'DejaVu Sans']
        math = dict(**{'mathtext.fontset': 'custom', 'mathtext.rm': font, 'mathtext.it': font + ':italic',
                       'mathtext.bf': font + ':bold', 'mathtext.default': 'it'})
    except ValueError:
        fam = ['DejaVu Sans']
        math = {'mathtext.fontset': 'dejavusans', 'mathtext.default': 'it'}
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',      # TrueType, editable text in the PDF
        'font.family': fam, 'font.size': fontsize, 'axes.linewidth': 0.7, 'savefig.facecolor': 'white',
        'axes.unicode_minus': True, **math})


def vgradient(ax, y0, y1, x0, x1, c_edge, c_far, edge='bottom', z=1, n=64):
    """Vertical gradient; colour c_edge sits at the band edge (`edge` = side of the rectangle where the edge is)."""
    g = np.linspace(0, 1, n)[:, None]
    a, b = np.array(to_rgb(c_edge)), np.array(to_rgb(c_far))
    img = a[None, None, :] * (1 - g[..., None]) + b[None, None, :] * g[..., None]
    if edge == 'top':
        img = img[::-1]
    ax.imshow(img, extent=[x0, x1, y0, y1], origin='lower', aspect='auto', zorder=z, interpolation='bilinear')


def rounded(ax, xy, w, h, fc, ec='none', lw=0.6, r=0.02, z=0, transform=None):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle='round,pad=0,rounding_size=%g' % r, fc=fc, ec=ec, lw=lw, zorder=z,
                                transform=transform or ax.transAxes))


def spin_arrow(ax, xc, y, color, direction):
    """Spin arrow centred on the level bar: 12 pt long, filled head, independent of the axis scale."""
    sgn = 1 if direction == 'up' else -1
    tip = offset_copy(ax.transData, fig=ax.figure, y=sgn * 6.2, units='points')
    ax.annotate('', xy=(xc, y), xycoords=tip, xytext=(0, -sgn * 12.4), textcoords='offset points', zorder=6,
                arrowprops=dict(arrowstyle='-|>', color=color, lw=1.25, shrinkA=0, shrinkB=0, mutation_scale=6.5,
                                joinstyle='miter'))


# ----------------------------------------------------------------------------------------------- layout
def _as_tuple(x):
    return tuple(x) if isinstance(x, (tuple, list)) else (x,)


def layout(spec):
    """x positions of every bar: pos[(channel, level, sub)] = (x0, x1); also channel width W and label x positions."""
    g = GEOM
    widths = []
    for ch in spec['channels']:
        w = 0.0
        for k, lv in enumerate(ch['levels']):
            n = len(_as_tuple(lv['e']))
            w += (g['group_gap'] if k else 0.0) + n * g['bar'] + (n - 1) * g['sub_gap']
        widths.append(w)
    W = max(widths) + g['left'] + g['right']
    pos, lab = {}, {}
    for c, ch in enumerate(spec['channels']):
        x = c * W + g['left']
        for k, lv in enumerate(ch['levels']):
            if k:
                x += g['group_gap']
            lab[(c, k)] = x - 0.13
            for s in range(len(_as_tuple(lv['e']))):
                pos[(c, k, s)] = (x, x + g['bar'])
                x += g['bar'] + g['sub_gap']
            x -= g['sub_gap']
    return pos, lab, W


def draw_bar(ax, xs, e, zero, color, direction, occ, half_arrow, fmt, show_value=True):
    y = round(e - zero, 2)
    y = 0.0 if abs(y) < 5e-3 else y
    ax.plot(xs, [y, y], color=INK, lw=2.1, solid_capstyle='round', zorder=5)
    xc = 0.5 * (xs[0] + xs[1])
    if occ == 1 or (occ == 0.5 and half_arrow):
        spin_arrow(ax, xc, y, color, direction)
    if show_value:
        ax.annotate(fmt % y, (xc, y), xytext=(0, -8.5), textcoords='offset points', ha='center', va='top',
                    fontsize=6.4, color=MUTED, zorder=7, path_effects=HALO)
    if occ == 0.5:
        ax.annotate(u'½', (xs[1], y), xytext=(2.0, 2.2), textcoords='offset points', ha='left', va='bottom',
                    fontsize=6.0, color=color, zorder=7, path_effects=HALO)


# ----------------------------------------------------------------------------------------------- one panel
def draw_scheme(ax, spec, ymax=None, ylabel=False, fmt='%.2f', ylo=YLO, ytick=2):
    zero = spec.get('zero', 0.0)
    vbm, cbm = spec.get('vbm', 0.0) - zero, spec['cbm'] - zero
    pos, lab, W = layout(spec)
    nch = len(spec['channels'])
    has_br = bool(spec.get('bracket'))
    xmax = nch * W + (1.0 if has_br else 0.2)
    if ymax is None:
        top = max(max(_as_tuple(lv['e'])) - zero for ch in spec['channels'] for lv in ch['levels'])
        ymax = max(cbm, top) + 1.6
    ax.set_xlim(-0.75, xmax)
    ax.set_ylim(ylo, ymax)
    ax.set_facecolor('#FBFCFE')
    for s in ('top', 'right', 'bottom'):
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_color(FAINT)
    ax.set_xticks([])
    ax.set_yticks(np.arange(np.ceil(ylo / ytick) * ytick, ymax + 1e-9, ytick))
    ax.tick_params(axis='y', labelsize=7, length=2.5, pad=1.5, color=FAINT)
    if ylabel:
        ax.set_ylabel('Energy (eV)', fontsize=8, labelpad=2)
    # band continua
    vgradient(ax, ylo, vbm, -0.75, xmax, BAND_EDGE, BAND_FAR, edge='top', z=1)
    vgradient(ax, cbm, ymax, -0.75, xmax, BAND_EDGE, BAND_FAR, edge='bottom', z=1)
    for yb in (vbm, cbm):
        ax.axhline(yb, color=BAND_LINE, lw=0.9, zorder=2)
    ax.text(xmax - 0.1, ymax - 0.12, 'CB', ha='right', va='top', fontsize=7.4, fontweight='bold', color=BAND_TXT, zorder=3)
    ax.text(xmax - 0.1, ylo + 0.1, 'VB', ha='right', va='bottom', fontsize=7.4, fontweight='bold', color=BAND_TXT, zorder=3)
    # channels: header, divider, bars, labels
    for c, ch in enumerate(spec['channels']):
        col, dr = ch.get('color', INK), ch.get('direction', 'up')
        xmid = c * W + W / 2 - 0.1
        if ch.get('name'):
            ax.text(xmid, ymax - 0.12, ch['name'], ha='center', va='top', fontsize=7.2, fontweight='bold', color=col, zorder=3)
        if c:
            ax.plot([c * W] * 2, [vbm + 0.25, cbm - 0.25], color=HAIR, lw=0.8, ls=(0, (1.5, 2.5)), zorder=2)
        for k, lv in enumerate(ch['levels']):
            es = _as_tuple(lv['e'])
            occ = lv.get('occ', 0)
            occ = tuple(occ) if isinstance(occ, (tuple, list)) else (occ,) * len(es)
            half = [s for s in range(len(es)) if occ[s] == 0.5]
            arrow_on = min(half, key=lambda s: (es[s], -s)) if half else None   # lowest 0.5 bar (right one if degenerate)
            for s, e in enumerate(es):
                draw_bar(ax, pos[(c, k, s)], e, zero, col, dr, occ[s], s == arrow_on, fmt)
            if lv.get('label'):
                ax.text(lab[(c, k)], es[0] - zero, lv['label'], ha='right', va='center', fontsize=9, color=INK, zorder=6)
    # dashed excitation arrow
    if spec.get('excitation'):
        (c1, k1, s1), (c2, k2, s2) = spec['excitation']
        e1 = _as_tuple(spec['channels'][c1]['levels'][k1]['e'])[s1] - zero
        e2 = _as_tuple(spec['channels'][c2]['levels'][k2]['e'])[s2] - zero
        col = spec['channels'][c2].get('color', RED)
        ax.annotate('', xy=(pos[(c2, k2, s2)][0] - 0.06, e2 - 0.02), xytext=(pos[(c1, k1, s1)][1] + 0.06, e1 + 0.02), zorder=4,
                    arrowprops=dict(arrowstyle='-|>', color=col, lw=0.8, ls=(0, (2.2, 1.6)), alpha=0.85, shrinkA=0, shrinkB=0,
                                    mutation_scale=5.5, connectionstyle='arc3,rad=-0.30'))
    # dimension bracket between two levels (drawn at the right-hand edge of the panel)
    if has_br:
        b = spec['bracket']
        (ca, ka, sa), (cb_, kb, sb) = b['a'], b['b']
        ya = _as_tuple(spec['channels'][ca]['levels'][ka]['e'])[sa] - zero
        yb_ = _as_tuple(spec['channels'][cb_]['levels'][kb]['e'])[sb] - zero
        xb = nch * W
        for (cc, kk, ss), yy in (((ca, ka, sa), ya), ((cb_, kb, sb), yb_)):
            ax.plot([pos[(cc, kk, ss)][1] + 0.1, xb + 0.12], [yy, yy], color=FAINT, lw=0.7, ls=(0, (1.5, 2)), zorder=3)
        ax.annotate('', xy=(xb, yb_), xytext=(xb, ya), zorder=6,
                    arrowprops=dict(arrowstyle='<|-|>', color=INK, lw=0.9, shrinkA=0, shrinkB=0, mutation_scale=5.5))
        ax.text(xb + 0.30, 0.5 * (ya + yb_), b['text'], rotation=90, ha='left', va='center', fontsize=6.8, color=INK,
                zorder=7, path_effects=HALO)
    if spec.get('term'):
        ax.text(0.035, 0.965, spec['term'], transform=ax.transAxes, ha='left', va='top', fontsize=10, color=MUTED, zorder=6)
    return ax


# ----------------------------------------------------------------------------------------------- grid of panels
def legend_strip(fig, y=0.03, x0=0.02, fs=7.0):
    """One-line key (figure coordinates): spin arrows, 1/2 occupation, excitation arrow, band continua."""
    base = fig.add_axes([0, 0, 1, 1], zorder=-5)
    base.axis('off'); base.set_xlim(0, 1); base.set_ylim(0, 1)
    r = fig.canvas.get_renderer()

    def text(x, t):
        tt = fig.text(x, y, t, ha='left', va='center', fontsize=fs, color=MUTED)
        return tt.get_window_extent(r).transformed(fig.transFigure.inverted()).x1

    def vec(x, col):
        base.annotate('', xy=(x, y + 0.016), xytext=(x, y - 0.016),
                      arrowprops=dict(arrowstyle='-|>', color=col, lw=1.2, shrinkA=0, shrinkB=0, mutation_scale=6))
        return x + 0.008
    x = text(vec(x0, BLUE), 'spin-up electron')
    x = text(vec(x + 0.020, RED), 'spin-down electron')
    fig.text(x + 0.022, y, u'½', ha='left', va='center', fontsize=fs, color=RED)
    x = text(x + 0.036, 'fractional occupation 0.5')
    base.annotate('', xy=(x + 0.050, y), xytext=(x + 0.022, y),
                  arrowprops=dict(arrowstyle='-|>', color=RED, lw=0.8, ls=(0, (2.2, 1.6)), shrinkA=0, shrinkB=0, mutation_scale=5.5))
    x = text(x + 0.058, 'excitation')
    rounded(base, (x + 0.020, y - 0.010), 0.022, 0.020, BAND_EDGE, r=0.004, z=1)
    text(x + 0.047, 'VB / CB continua')


def scheme_grid(specs, col_titles=None, row_titles=None, subtitle=None, figsize=(7.0, 4.7), ymax=None,
                margins=dict(left=0.115, right=0.995, top=0.865, bottom=0.115), key=True, **kw):
    """specs[row][col] -> figure.  Titles go above the columns / left of the rows."""
    nr, nc = len(specs), len(specs[0])
    fig, axs = plt.subplots(nr, nc, figsize=figsize, sharey=True, squeeze=False)
    fig.subplots_adjust(wspace=0.10, hspace=0.07, **margins)
    for i in range(nr):
        for j in range(nc):
            draw_scheme(axs[i, j], specs[i][j], ymax=ymax, ylabel=(j == 0), **kw)
    for j in range(nc):
        p = axs[0, j].get_position()
        xc = 0.5 * (p.x0 + p.x1)
        if col_titles:
            fig.text(xc, margins['top'] + 0.090, col_titles[j], ha='center', va='center', fontsize=11.5, fontweight='bold', color=INK)
        if subtitle:
            fig.text(xc, margins['top'] + 0.050, subtitle, ha='center', va='center', fontsize=7.4, color=MUTED)
    for i in range(nr):
        if row_titles:
            p = axs[i, 0].get_position()
            fig.text(0.014, 0.5 * (p.y0 + p.y1), row_titles[i], rotation=90, ha='center', va='center', fontsize=10,
                     fontweight='bold', color=INK)
    if key:
        legend_strip(fig, y=margins['bottom'] - 0.07)
    return fig, axs


# ----------------------------------------------------------------------------------------------- example (NV- in diamond, Fig. 5)
def nv_panel(state, ua1, ue, da1, de, cbm, dEa1e=None):
    """Ground (3A2) or excited (3E) state of NV-;  energies relative to the VBM, spin-up/-down a1 and e levels."""
    gs = state == 'GS'
    dn_e = dict(label=r'$e$', e=de, occ=0 if gs else 0.5)
    spec = dict(vbm=0.0, cbm=cbm, term=r'$^{3}A_{2}$' if gs else r'$^{3}E$',
                channels=[dict(name='spin-up', color=BLUE, direction='up',
                               levels=[dict(label=r'$a_1$', e=ua1, occ=1), dict(label=r'$e$', e=ue, occ=1)]),
                          dict(name='spin-down', color=RED, direction='down',
                               levels=[dict(label=r'$a_1$', e=da1, occ=1 if gs else 0), dn_e])])
    if gs and dEa1e is not None:
        spec['bracket'] = dict(a=(1, 0, 0), b=(1, 1, 0), text=r'$\Delta E_{a_1e}$ = %.2f eV' % dEa1e)
    if not gs:
        spec['excitation'] = ((1, 0, 0), (1, 1, 0))
    return spec


EXAMPLE = {   # eigenvalues (eV) relative to the VBM, as in the paper's Fig. 5 (read from the Kohn-Sham level plots)
    'PBE': dict(GS=(.83, (2.02, 2.02), 1.38, (3.26, 3.26), 4.20), ES=(.58, (2.29, 2.29), 1.35, (3.28, 3.28), 4.19)),
    'HSE': dict(GS=(.67, (1.88, 1.88), 1.63, (4.91, 4.91), 5.75), ES=(.41, (2.11, 2.11), 2.66, (4.34, 4.34), 5.74)),
    'CAM': dict(GS=(.69, (1.96, 1.96), 1.69, (6.33, 6.33), 7.38), ES=(.45, (2.20, 2.20), 4.04, (5.11, 5.11), 7.37))}

if __name__ == '__main__':
    setup()
    funcs = ['PBE', 'HSE', 'CAM']
    specs = [[nv_panel('ES', *EXAMPLE[f]['ES']) for f in funcs],
             [nv_panel('GS', *EXAMPLE[f]['GS'], dEa1e=EXAMPLE[f]['GS'][3][0] - EXAMPLE[f]['GS'][2]) for f in funcs]]
    fig, _ = scheme_grid(specs, col_titles=funcs, row_titles=['Excited state', 'Ground state'],
                         subtitle=u'4×4×4 supercell, Γ point', ymax=8.6)
    fig.savefig('level_scheme_example.pdf')          # fixed canvas: the PDF page is the printed size
    fig.savefig('level_scheme_example.png', dpi=200)
    print('wrote level_scheme_example.pdf / .png')
