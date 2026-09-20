"""
dijkstra_gcode_planner.py
==========================

STL (layer piatto, senza spessore, esportato da Blender)
    -> profilo esterno + fori (trimesh)
    -> grafo di visibilita' tra i punti campionati sui contorni (shapely)
    -> Dijkstra tra il punto min e il punto max del contorno esterno
    -> G-code del travel move risultante
    -> immagine PNG col percorso trovato

Uso:
    python dijkstra_gcode_planner.py layer_piatto.stl
    (senza argomenti, cerca "layer_piatto.stl" nella stessa cartella dello script)

Dipendenze: pip install -r requirements.txt  (numpy, shapely, matplotlib, trimesh, rtree, mapbox_earcut)
"""

from __future__ import annotations

import heapq
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from shapely.geometry import LineString, Polygon

Coord = Tuple[float, float]


# ---------------------------------------------------------------------------
# Grafo di visibilità + Dijkstra
# ---------------------------------------------------------------------------

@dataclass
class VisibilityGraph:
    outer: Polygon                 # perimetro esterno (non si può uscire)
    holes: List[Polygon]           # fori (non si possono attraversare)
    nodes: Dict[str, Coord] = field(default_factory=dict)
    adj: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)
    contour_edges: set = field(default_factory=set)

    def add_node(self, name: str, coord: Coord) -> None:
        self.nodes[name] = coord
        self.adj.setdefault(name, [])

    def add_contour(self, names: List[str]) -> None:
        """Punti consecutivi sullo stesso contorno (profilo già stampato):
        sempre collegabili, anche se la corda taglia leggermente il foro."""
        n = len(names)
        for i in range(n):
            a, b = names[i], names[(i + 1) % n]
            self.contour_edges.add(frozenset((a, b)))

    def _segment_is_valid(self, p1: Coord, p2: Coord) -> bool:
        seg = LineString([p1, p2])
        if not self.outer.buffer(1e-6).contains(seg):
            return False
        for hole in self.holes:
            if hole.intersection(seg).length > 1e-9:
                return False
        return True

    def build_edges(self) -> None:
        names = list(self.nodes.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                pa, pb = self.nodes[a], self.nodes[b]
                if frozenset((a, b)) in self.contour_edges or self._segment_is_valid(pa, pb):
                    d = math.dist(pa, pb)
                    self.adj[a].append((b, d))
                    self.adj[b].append((a, d))

    def dijkstra(self, start: str, goal: str) -> Tuple[List[str], float]:
        dist = {n: math.inf for n in self.nodes}
        prev: Dict[str, str] = {}
        dist[start] = 0.0
        pq = [(0.0, start)]
        visited = set()
        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            if u == goal:
                break
            for v, w in self.adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        if dist[goal] == math.inf:
            return [], math.inf
        path = [goal]
        while path[-1] != start:
            path.append(prev[path[-1]])
        path.reverse()
        return path, dist[goal]


# ---------------------------------------------------------------------------
# Lettura della mesh STL (layer piatto) e costruzione del grafo
# ---------------------------------------------------------------------------

def _decimate_if_needed(coords, max_points: int) -> List[Coord]:
    """Di norma NON decima: tiene tutti i punti del contorno. Riduce solo
    se il contorno ha più di max_points punti (mesh molto fitte), giusto
    per non far esplodere il numero di coppie testate in build_edges()."""
    pts = list(coords)
    if len(pts) > 1 and math.dist(pts[0], pts[-1]) < 1e-9:
        pts = pts[:-1]
    if len(pts) <= max_points:
        return pts
    step = max(1, len(pts) // max_points)
    return pts[::step]


def graph_from_flat_layer_mesh(stl_path: str, max_points_per_contour: int = 500) -> Tuple[VisibilityGraph, Dict[str, Coord]]:
    """Legge una mesh STL piatta (layer singolo, z=0) e ne ricava il
    contorno esterno + eventuali fori.

    Non uso mesh.outline()/polygons_full (si basano sulla topologia
    edge-facce della mesh: bastano poche facce sovrapposte, tipiche dei
    "ventagli" di triangolazione che Blender genera riempiendo un buco
    concavo, per corrompere il calcolo e perdere interi contorni).
    Uso invece l'unione geometrica 2D di tutti i triangoli
    (shapely.ops.unary_union): il risultato dipende solo dall'area
    coperta da ciascun triangolo, quindi è immune a facce sovrapposte
    o a una topologia mesh malformata.
    """
    import trimesh
    from shapely.ops import unary_union

    mesh = trimesh.load(stl_path)
    tris2d = mesh.vertices[:, :2][mesh.faces]
    tri_polys = [Polygon(t) for t in tris2d]
    tri_polys = [p for p in tri_polys if p.is_valid and p.area > 1e-9]
    if not tri_polys:
        raise ValueError("Nessun triangolo valido nella mesh")

    merged = unary_union(tri_polys)
    if merged.geom_type == "MultiPolygon":
        merged = max(merged.geoms, key=lambda p: p.area)
    if merged.geom_type != "Polygon":
        raise ValueError(f"Forma inattesa dopo l'unione dei triangoli: {merged.geom_type}")

    outer_pts = _decimate_if_needed(list(merged.exterior.coords), max_points_per_contour)
    hole_rings = [_decimate_if_needed(list(h.coords), max_points_per_contour) for h in merged.interiors]

    vg = VisibilityGraph(outer=Polygon(merged.exterior.coords),
                          holes=[Polygon(h.coords) for h in merged.interiors])
    labeled: Dict[str, Coord] = {}

    outer_names = []
    for i, p in enumerate(outer_pts):
        name = f"outer_{i}"
        vg.add_node(name, tuple(p))
        labeled[name] = tuple(p)
        outer_names.append(name)
    vg.add_contour(outer_names)

    for hi, ring in enumerate(hole_rings):
        names = []
        for i, p in enumerate(ring):
            name = f"hole{hi}_{i}"
            vg.add_node(name, tuple(p))
            labeled[name] = tuple(p)
            names.append(name)
        vg.add_contour(names)

    vg.build_edges()
    return vg, labeled


# ---------------------------------------------------------------------------
# G-code del travel move
# ---------------------------------------------------------------------------

def path_to_gcode(path_coords: List[Coord], feedrate_travel: int = 6000,
                   retract_mm: float = 0.6, retract_feedrate: int = 2400) -> List[str]:
    lines = [f"; --- travel move ottimizzato (Dijkstra, {len(path_coords)} nodi) ---",
             f"G1 E-{retract_mm:.3f} F{retract_feedrate}  ; retract"]
    for x, y in path_coords:
        lines.append(f"G0 X{x:.3f} Y{y:.3f} F{feedrate_travel}")
    lines.append(f"G1 E{retract_mm:.3f} F{retract_feedrate}  ; unretract")
    lines.append("; --- fine travel move ---")
    return lines


# ---------------------------------------------------------------------------
# Immagine PNG del risultato
# ---------------------------------------------------------------------------

def plot_result(vg: VisibilityGraph, labeled: Dict[str, Coord], path: List[str],
                 dist: float, fname: str, elapsed_s: float = None) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ox, oy = vg.outer.exterior.xy
    ax.plot(ox, oy, "k-", lw=2, label="Outer perimeter")
    for hole in vg.holes:
        hx, hy = hole.exterior.xy
        ax.plot(hx, hy, "k-", lw=2)
        ax.fill(hx, hy, color="0.85")

    for x, y in labeled.values():
        ax.plot(x, y, "rx", ms=10, mew=2)

    px = [labeled[n][0] for n in path]
    py = [labeled[n][1] for n in path]
    ax.plot(px, py, "b--o", lw=2, label="Optimized travel (Dijkstra)")
    ax.plot(px[0], py[0], "go", ms=12, label="Start")
    ax.plot(px[-1], py[-1], "mo", ms=12, label="Goal")

    ax.set_aspect("equal")
    title = f"Optimal distance: {dist:.2f} mm"
    if elapsed_s is not None:
        # sotto 1 s mostro in millisecondi, altrimenti in secondi
        time_str = f"{elapsed_s*1000:.1f} ms" if elapsed_s < 1 else f"{elapsed_s:.3f} s"
        title += f"  |  Computation time: {time_str}  ({len(vg.nodes)} nodes)"
    ax.legend(title=title, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2)
    ax.set_title("Optimized travel move (visibility graph + Dijkstra)")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stl_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(script_dir, "layer_piatto.stl")

    if not os.path.exists(stl_path):
        print(f"File non trovato: {stl_path}")
        print("Esporta da Blender uno STL piatto con questo nome (o passalo come argomento).")
        sys.exit(1)

    # cronometro: dalla costruzione del grafo (compresa lettura STL) fino
    # alla fine della ricerca di Dijkstra, cioè tutto il tempo di calcolo
    # necessario per arrivare al percorso ottimale
    t0 = time.perf_counter()
    vg, labeled = graph_from_flat_layer_mesh(stl_path)

    outer_names = [n for n in vg.nodes if n.startswith("outer_")]
    start = min(outer_names, key=lambda n: labeled[n])
    goal = max(outer_names, key=lambda n: labeled[n])
    path, dist = vg.dijkstra(start, goal)
    elapsed = time.perf_counter() - t0

    direct = math.dist(labeled[start], labeled[goal])
    print(f"File: {stl_path}")
    print(f"Nodi nel grafo: {len(vg.nodes)}")
    print(f"start={start} {labeled[start]}   goal={goal} {labeled[goal]}")
    print(f"Distanza diretta: {direct:.2f} mm")
    print(f"Percorso trovato: {' -> '.join(path)}")
    print(f"Lunghezza travel ottimizzato: {dist:.2f} mm")
    print(f"Tempo di calcolo (grafo + Dijkstra): {elapsed*1000:.1f} ms\n")

    print("\n".join(path_to_gcode([labeled[n] for n in path])))

    out_png = os.path.join(script_dir, "dijkstra_path_result.png")
    plot_result(vg, labeled, path, dist, out_png, elapsed_s=elapsed)
    print(f"\nImmagine salvata in: {out_png}")


if __name__ == "__main__":
    main()