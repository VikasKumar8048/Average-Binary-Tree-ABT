"""
abt/viz/visualizer.py
=====================
Visualization engine for ABT trees.

Provides:
  1. ASCII tree printer
  2. Graphviz DOT export
  3. NetworkX graph export
  4. Matplotlib static plot
  5. Plotly interactive plot

Reference: Paper Section VI (figures 1 and 2).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Tuple

from abt.core import Node, bfs_values, height as tree_height, node_count


# ---------------------------------------------------------------------------
# 1. ASCII Tree
# ---------------------------------------------------------------------------

def ascii_tree(root: Optional[Node], label: str = "") -> str:
    """Return a multi-line ASCII representation of a binary tree.

    Each level is printed on one line with values separated by spaces.
    """
    if label:
        lines = [f"=== {label} ==="]
    else:
        lines = []
    if root is None:
        lines.append("(empty)")
        return "\n".join(lines)

    q: deque = deque([root])
    level = 0
    while q:
        row = []
        for _ in range(len(q)):
            n = q.popleft()
            row.append(f"{n.val:g}")
            if n.left:
                q.append(n.left)
            if n.right:
                q.append(n.right)
        indent = "  " * (tree_height(root) - level)
        lines.append(f"L{level}: {indent}{('  ' * level).join(row)}")
        level += 1
    return "\n".join(lines)


def side_by_side_ascii(T: Optional[Node], T_prime: Optional[Node]) -> str:
    """Print T and T' side-by-side for comparison."""
    left_lines  = ascii_tree(T,       "T  (original)").split("\n")
    right_lines = ascii_tree(T_prime, "T' (ABT)").split("\n")
    width = max((len(l) for l in left_lines), default=0) + 4
    max_rows = max(len(left_lines), len(right_lines))
    result = []
    for i in range(max_rows):
        l = left_lines[i]  if i < len(left_lines)  else ""
        r = right_lines[i] if i < len(right_lines) else ""
        result.append(l.ljust(width) + r)
    return "\n".join(result)


# ---------------------------------------------------------------------------
# 2. Graphviz DOT export
# ---------------------------------------------------------------------------

def to_dot(
    root: Optional[Node],
    graph_name: str = "ABT",
    node_color: str = "lightblue",
    leaf_color: str = "lightyellow",
) -> str:
    """Export a tree as a Graphviz DOT string.

    Parameters
    ----------
    root : Node or None
    graph_name : str
        Name of the digraph.
    node_color : str
        Fill colour for internal nodes.
    leaf_color : str
        Fill colour for leaf nodes.

    Returns
    -------
    str
        Valid DOT language string that can be saved to a .dot file.
    """
    lines = [f'digraph {graph_name} {{', '  node [shape=circle, style=filled];']

    if root is None:
        lines.append("}")
        return "\n".join(lines)

    counter = [0]

    def _node_id(n: Node) -> str:
        return f"n{id(n)}"

    q: deque = deque([root])
    while q:
        n = q.popleft()
        nid = _node_id(n)
        color = leaf_color if n.is_leaf() else node_color
        label = f"{n.val:g}"
        lines.append(f'  {nid} [label="{label}", fillcolor="{color}"];')
        if n.left:
            lines.append(f"  {nid} -> {_node_id(n.left)};")
            q.append(n.left)
        if n.right:
            lines.append(f"  {nid} -> {_node_id(n.right)};")
            q.append(n.right)

    lines.append("}")
    return "\n".join(lines)


def save_dot(root: Optional[Node], path: str, graph_name: str = "ABT") -> None:
    """Save DOT string to *path*."""
    dot = to_dot(root, graph_name=graph_name)
    with open(path, "w") as f:
        f.write(dot)
    print(f"DOT file saved: {path}")


# ---------------------------------------------------------------------------
# 3. NetworkX graph export
# ---------------------------------------------------------------------------

def to_networkx(root: Optional[Node]):
    """Export a tree to a NetworkX DiGraph.

    Requires networkx.  Node attributes include 'val', 'is_leaf', 'depth'.

    Returns
    -------
    nx.DiGraph or None
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx is required for to_networkx(). pip install networkx")

    G = nx.DiGraph()
    if root is None:
        return G

    q: deque = deque([(root, 0)])
    while q:
        n, d = q.popleft()
        G.add_node(
            id(n),
            val=n.val,
            label=f"{n.val:g}",
            is_leaf=n.is_leaf(),
            depth=d,
        )
        if n.left:
            G.add_edge(id(n), id(n.left), side="left")
            q.append((n.left, d + 1))
        if n.right:
            G.add_edge(id(n), id(n.right), side="right")
            q.append((n.right, d + 1))

    return G


# ---------------------------------------------------------------------------
# 4. Matplotlib static plot
# ---------------------------------------------------------------------------

def _compute_positions(root: Optional[Node]) -> Dict[int, Tuple[float, float]]:
    """Compute (x, y) positions for tree drawing.

    y is determined by depth; x by in-order traversal rank.
    """
    positions: Dict[int, Tuple[float, float]] = {}
    counter = [0]
    h = tree_height(root)

    def _inorder(n: Optional[Node], depth: int) -> None:
        if n is None:
            return
        _inorder(n.left, depth + 1)
        positions[id(n)] = (float(counter[0]), float(h - depth))
        counter[0] += 1
        _inorder(n.right, depth + 1)

    _inorder(root, 0)
    return positions


def plot_tree(
    root: Optional[Node],
    title: str = "ABT",
    ax=None,
    node_color: str = "skyblue",
    leaf_color: str = "lightyellow",
    show: bool = True,
):
    """Plot a binary tree using matplotlib.

    Parameters
    ----------
    root : Node or None
    title : str
    ax : matplotlib Axes or None
        If None, a new figure is created.
    node_color, leaf_color : str
    show : bool
        Call plt.show() if True.

    Returns
    -------
    matplotlib Figure
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        raise ImportError("matplotlib is required. pip install matplotlib")

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    if root is None:
        ax.text(0.5, 0.5, "(empty)", ha="center", va="center")
        ax.set_title(title)
        if show and fig:
            plt.show()
        return fig

    pos = _compute_positions(root)
    q: deque = deque([root])

    # Draw edges first
    while q:
        n = q.popleft()
        x, y = pos[id(n)]
        if n.left:
            xl, yl = pos[id(n.left)]
            ax.plot([x, xl], [y, yl], "k-", linewidth=1, zorder=1)
            q.append(n.left)
        if n.right:
            xr, yr = pos[id(n.right)]
            ax.plot([x, xr], [y, yr], "k-", linewidth=1, zorder=1)
            q.append(n.right)

    # Draw nodes
    q = deque([root])
    while q:
        n = q.popleft()
        x, y = pos[id(n)]
        color = leaf_color if n.is_leaf() else node_color
        circle = plt.Circle((x, y), 0.35, color=color, ec="black", zorder=2)
        ax.add_patch(circle)
        ax.text(x, y, f"{n.val:g}", ha="center", va="center", fontsize=8, zorder=3)
        if n.left:
            q.append(n.left)
        if n.right:
            q.append(n.right)

    ax.autoscale()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)

    if show and fig:
        plt.tight_layout()
        plt.show()

    return fig


def plot_transformation(
    T: Optional[Node],
    T_prime: Optional[Node],
    title: str = "ABT Transformation",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Plot T and T' side by side, showing the ABT transformation.

    Reproduces the visual style of Figures 1 and 2 in the paper.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required. pip install matplotlib")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plot_tree(T,       "T (original)", ax=ax1, show=False,
              node_color="cornflowerblue", leaf_color="orange")
    plot_tree(T_prime, "T' (ABT)",     ax=ax2, show=False,
              node_color="mediumseagreen", leaf_color="mediumseagreen")
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved: {save_path}")

    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# 5. Plotly interactive plot
# ---------------------------------------------------------------------------

def plot_interactive(
    root: Optional[Node],
    title: str = "ABT Interactive",
    save_html: Optional[str] = None,
):
    """Create an interactive Plotly tree visualization.

    Parameters
    ----------
    root : Node or None
    title : str
    save_html : str or None
        If provided, save the figure as an HTML file.

    Returns
    -------
    plotly Figure or None
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly is required. pip install plotly")

    if root is None:
        print("(empty tree — nothing to plot)")
        return None

    pos = _compute_positions(root)
    edge_x, edge_y = [], []
    node_x, node_y, node_text, node_color_list = [], [], [], []

    q: deque = deque([root])
    while q:
        n = q.popleft()
        x, y = pos[id(n)]
        for child in (n.left, n.right):
            if child:
                xc, yc = pos[id(child)]
                edge_x += [x, xc, None]
                edge_y += [y, yc, None]
                q.append(child)

    q = deque([root])
    while q:
        n = q.popleft()
        x, y = pos[id(n)]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{n.val:g}")
        node_color_list.append("#98c1d9" if not n.is_leaf() else "#ffd166")
        if n.left:
            q.append(n.left)
        if n.right:
            q.append(n.right)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1, color="#888"),
        hoverinfo="none",
    )
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="middle center",
        marker=dict(size=30, color=node_color_list, line=dict(width=1, color="black")),
        hoverinfo="text",
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=title,
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            hovermode="closest",
        ),
    )

    if save_html:
        fig.write_html(save_html)
        print(f"Interactive HTML saved: {save_html}")

    return fig
