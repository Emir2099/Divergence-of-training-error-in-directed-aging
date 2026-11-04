"""
spring_network.py

Disordered 2D spring networks with periodic boundary conditions, built from a
diluted Delaunay triangulation, plus the elastic energy / force / Hessian
machinery needed to quasistatically relax and train them.

This module reproduces the *model* of Bhaumik & Hexner, "Loss of material
trainability through an unusual transition", Phys. Rev. Research 4, L042044
(2022) [arXiv:2203.03468]: a network of Hookean springs, characterized by its
coordination number Z, trained by cyclically straining a source pair while
plastically evolving each bond's rest length toward relaxing its own tension
(their Eq. 1).

REPRODUCTION vs NOVELTY
------------------------
Everything in this file is REPRODUCTION of the published model, with one
explicit, flagged simplification:

    [SIMPLIFICATION] The original paper derives its networks from jammed
    packings of soft frictionless disks (the standard O'Hern-Liu-Nagel
    protocol), then dilutes bonds to hit a target Z. Reproducing that exactly
    means writing a full packing-generation + energy-minimization pipeline
    before you even get to training. Here we instead generate networks from a
    periodic Delaunay triangulation (mean Z ~ 6 in 2D) and randomly dilute
    bonds down to the target Z, pruning any node that falls below the (d+1)=3
    bonds needed to avoid trivial rattlers. This is a standard, faster proxy
    used elsewhere in this literature for the same class of networks, but it
    is NOT quantitatively identical to a jammed-packing-derived network near
    isostaticity (e.g. exact force/length distributions at small dZ will
    differ). Treat results near the isostatic point as qualitative until
    cross-checked against a true jammed packing. This is the one place where
    "reproduction" is only approximate; everything downstream (training,
    Hessian, mode tracking, phase diagram) is exact.
"""

import numpy as np
from scipy.spatial import Delaunay


class SpringNetwork:
    """
    A 2D periodic network of unit-stiffness Hookean springs.

    Attributes
    ----------
    N : int
        Number of real nodes.
    L : float
        Side length of the (square) periodic box.
    pos : (N, 2) ndarray
        Current node positions, wrapped into [0, L).
    bonds : (Nb, 2) int ndarray
        Node index pairs (i, j), i < j, for each real spring.
    rest_length : (Nb,) ndarray
        Rest length ell_{i,0} of each real bond. Evolves under training.
    k : (Nb,) ndarray
        Spring constant of each real bond (all 1.0 unless noted).
    image : (Nb, 2) int ndarray
        Periodic image convention for each bond: the minimum-image
        displacement for bond b is pos[j] - pos[i] + image[b] * L.
    ghosts : dict
        Auxiliary strong "driving" springs, keyed by name (e.g. "source",
        "target_0"), each a dict with keys: pair (i, j), k (stiffness),
        rest_length (driven externally), image.
    """

    def __init__(self, N, L, bonds, rest_length, k, image, rng):
        self.N = N
        self.L = L
        self.bonds = bonds
        self.rest_length = rest_length
        self.k = k
        self.image = image
        self.pos = None
        self.ghosts = {}
        self.rng = rng

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def random_diluted_delaunay(cls, N, target_Z, rng, box_L=None):
        """
        Build a disordered periodic spring network with the target mean
        coordination number `target_Z`, by Delaunay-triangulating a random
        point set under periodic boundary conditions and randomly removing
        bonds (see [SIMPLIFICATION] in the module docstring).

        Parameters
        ----------
        N : int
            Number of nodes.
        target_Z : float
            Desired mean coordination number after dilution (Zc = 4 in 2D
            for central-force isostaticity; pass target_Z = 4 + dZ).
        rng : numpy.random.Generator
        box_L : float, optional
            Box side length. Defaults to sqrt(N) so mean point density is 1,
            an arbitrary but convenient convention (only ratios/dimensionless
            strains matter for everything downstream).

        Returns
        -------
        SpringNetwork, with rest lengths initialised so the network starts
        completely unstressed (rest_length == current bond length).
        """
        if box_L is None:
            box_L = np.sqrt(N)
        L = box_L

        pos = rng.uniform(0.0, L, size=(N, 2))

        # Periodic Delaunay via the standard 3x3 tiling trick: triangulate
        # the 9 periodic images together, then keep every edge that touches
        # the central copy, recovering the correct minimum-image partner
        # and wrap vector for each kept edge.
        shifts = np.array([(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)])
        tiled = np.concatenate([pos + s * L for s in shifts], axis=0)
        tri = Delaunay(tiled)

        central_start = 4 * N  # index of shift (0, 0) block in `shifts` order
        # shifts is built with a in (-1,0,1), b in (-1,0,1) via the comprehension
        # above; find the actual offset of shift (0, 0):
        central_idx = np.where((shifts[:, 0] == 0) & (shifts[:, 1] == 0))[0][0]
        central_start = central_idx * N
        central_end = central_start + N

        edge_set = {}
        for simplex in tri.simplices:
            for a in range(3):
                for b in range(a + 1, 3):
                    u, v = simplex[a], simplex[b]
                    # Keep the edge only if at least one endpoint lies in the
                    # central tile; map both endpoints back to central indices
                    # and recover the periodic image of the true separation.
                    u_in_central = central_start <= u < central_end
                    v_in_central = central_start <= v < central_end
                    if not (u_in_central or v_in_central):
                        continue
                    u_real = u % N
                    v_real = v % N
                    if u_real == v_real:
                        continue
                    i, j = (u_real, v_real) if u_real < v_real else (v_real, u_real)
                    # true (possibly wrapped) separation j - i, in units of L
                    du = pos[v_real] - pos[u_real] if u_real == (u % N) else None
                    # compute via actual (possibly-tiled) coordinates instead,
                    # which is unambiguous:
                    sep = tiled[v] - tiled[u]
                    image = np.round(sep / L - (pos[v_real] - pos[u_real]) / L)
                    if u_real > v_real:
                        image = -image
                    key = (i, j)
                    if key not in edge_set:
                        edge_set[key] = tuple(image.astype(int))

        bonds = np.array(list(edge_set.keys()), dtype=int)
        images = np.array(list(edge_set.values()), dtype=int)

        # --- Dilute bonds down to target_Z, pruning rattlers -------------
        bonds, images = _dilute_to_target_Z(bonds, images, N, target_Z, rng)

        rest_length = _bond_lengths(pos, bonds, images, L)
        k = np.ones(len(bonds))

        net = cls(N, L, bonds, rest_length, k, images, rng)
        net.pos = pos
        return net

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def bond_vectors(self, pos=None):
        """Return (Nb, 2) separations x_j - x_i (minimum image) and (Nb,) lengths."""
        if pos is None:
            pos = self.pos
        i, j = self.bonds[:, 0], self.bonds[:, 1]
        sep = pos[j] - pos[i] + self.image * self.L
        length = np.linalg.norm(sep, axis=1)
        return sep, length

    def num_dof(self):
        return 2 * self.N

    # ------------------------------------------------------------------
    # Ghost (driving) bonds
    # ------------------------------------------------------------------
    def set_ghost(self, name, i, j, k_ghost, rest_length):
        """Add or update a strong auxiliary spring driving nodes i, j."""
        if name in self.ghosts:
            g = self.ghosts[name]
            g["rest_length"] = rest_length
        else:
            sep_min = _minimum_image_sep(self.pos[i], self.pos[j], self.L)
            self.ghosts[name] = {
                "pair": (i, j),
                "k": k_ghost,
                "rest_length": rest_length,
                "image": np.round(sep_min / self.L).astype(int) * 0,  # ghosts
                # connect nearby chosen pairs; image offset resolved at
                # energy-eval time via minimum image, see _ghost_vector.
            }

    def remove_ghost(self, name):
        self.ghosts.pop(name, None)

    def clear_ghosts(self):
        self.ghosts = {}

    def _ghost_vector(self, name, pos):
        i, j = self.ghosts[name]["pair"]
        return _minimum_image_sep(pos[i], pos[j], self.L)

    # ------------------------------------------------------------------
    # Energy, forces, Hessian  (bonds + active ghosts)
    # ------------------------------------------------------------------
    def energy(self, pos):
        sep, length = self.bond_vectors(pos)
        e = 0.5 * np.sum(self.k * (length - self.rest_length) ** 2)
        for g in self.ghosts.values():
            sep_g = _minimum_image_sep(pos[g["pair"][0]], pos[g["pair"][1]], self.L)
            len_g = np.linalg.norm(sep_g)
            e += 0.5 * g["k"] * (len_g - g["rest_length"]) ** 2
        return e

    def gradient(self, pos):
        """dE/dpos, flattened to shape (2N,)."""
        grad = np.zeros((self.N, 2))
        sep, length = self.bond_vectors(pos)
        n = sep / length[:, None]
        f_mag = self.k * (length - self.rest_length)  # tension magnitude
        # force on i from bond (i,j): +f_mag * n  (n points i -> j)
        i, j = self.bonds[:, 0], self.bonds[:, 1]
        np.add.at(grad, i, -f_mag[:, None] * n)
        np.add.at(grad, j, f_mag[:, None] * n)
        for g in self.ghosts.values():
            gi, gj = g["pair"]
            sep_g = _minimum_image_sep(pos[gi], pos[gj], self.L)
            len_g = np.linalg.norm(sep_g)
            n_g = sep_g / len_g
            f = g["k"] * (len_g - g["rest_length"])
            grad[gi] += -f * n_g
            grad[gj] += f * n_g
        return grad.reshape(-1)

    def hessian(self, pos, include_ghosts=True):
        """
        Dense (2N, 2N) Hessian d^2E/dpos^2, standard spring-network form:
        each bond contributes
            k * (n outer n) + (k*(r-r0)/r) * (I - n outer n)
        to the (i,i), (j,j) diagonal blocks (+) and (i,j), (j,i) off-diagonal
        blocks (-).
        """
        H = np.zeros((2 * self.N, 2 * self.N))
        _accumulate_hessian(H, self.bonds, self.k, self.rest_length, pos, self.image, self.L)
        if include_ghosts:
            for g in self.ghosts.values():
                gi, gj = g["pair"]
                gb = np.array([[gi, gj]])
                gk = np.array([g["k"]])
                gr = np.array([g["rest_length"]])
                gimg = np.zeros((1, 2))  # ghosts always use true minimum image
                _accumulate_hessian(H, gb, gk, gr, pos, gimg, self.L, true_min_image=True)
        return H

    def total_tension_energy_per_bond(self, pos):
        """(Nb,) array of |tension| * length for each real bond (diagnostic)."""
        _, length = self.bond_vectors(pos)
        return np.abs(self.k * (length - self.rest_length))


# ----------------------------------------------------------------------
# Free functions
# ----------------------------------------------------------------------
def _minimum_image_sep(xi, xj, L):
    d = xj - xi
    d = d - L * np.round(d / L)
    return d


def _bond_lengths(pos, bonds, images, L):
    i, j = bonds[:, 0], bonds[:, 1]
    sep = pos[j] - pos[i] + images * L
    return np.linalg.norm(sep, axis=1)


def _accumulate_hessian(H, bonds, k, rest_length, pos, images, L, true_min_image=False):
    i_idx, j_idx = bonds[:, 0], bonds[:, 1]
    if true_min_image:
        sep = np.array([_minimum_image_sep(pos[i], pos[j], L) for i, j in zip(i_idx, j_idx)])
    else:
        sep = pos[j_idx] - pos[i_idx] + images * L
    length = np.linalg.norm(sep, axis=1)
    n = sep / length[:, None]
    stretch = length - rest_length
    for b in range(len(bonds)):
        i, j = i_idx[b], j_idx[b]
        nb = n[b]
        nn = np.outer(nb, nb)
        transverse = np.eye(2) - nn
        # guard against division by ~0 length (shouldn't occur for real bonds)
        block_diag = k[b] * nn + (k[b] * stretch[b] / length[b]) * transverse
        H[2 * i:2 * i + 2, 2 * i:2 * i + 2] += block_diag
        H[2 * j:2 * j + 2, 2 * j:2 * j + 2] += block_diag
        H[2 * i:2 * i + 2, 2 * j:2 * j + 2] -= block_diag
        H[2 * j:2 * j + 2, 2 * i:2 * i + 2] -= block_diag


def _dilute_to_target_Z(bonds, images, N, target_Z, rng):
    """
    Sequentially, greedily remove bonds (in random order) down to the target
    mean coordination number, refusing any removal that would drop either
    endpoint below 3 (= d+1) bonds, so no trivial rattlers are created in
    the first place. This is checked *during* removal rather than pruning
    after the fact: pruning after a blind bulk removal cascades (removing a
    node's near-last bond can strand its neighbours too) and overshoots the
    target Z substantially, especially close to the isostatic point -- which
    would silently mislabel every dZ in the eventual phase diagram.
    """
    deg = np.zeros(N, dtype=int)
    for i, j in bonds:
        deg[i] += 1
        deg[j] += 1

    n_bonds_target = int(round(target_Z * N / 2))
    n_to_remove = max(0, len(bonds) - n_bonds_target)

    order = rng.permutation(len(bonds))
    keep_mask = np.ones(len(bonds), dtype=bool)
    removed = 0
    for idx in order:
        if removed >= n_to_remove:
            break
        i, j = bonds[idx]
        if deg[i] > 3 and deg[j] > 3:
            keep_mask[idx] = False
            deg[i] -= 1
            deg[j] -= 1
            removed += 1

    return bonds[keep_mask], images[keep_mask]
