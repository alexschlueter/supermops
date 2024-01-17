import itertools
import json

from jinja2 import Template, Environment
import numpy as np

from tex_utils import *

with open("plot_data.json", "r") as file:
    plot_data = json.load(file)

env = Environment(
    block_start_string="\BLOCK{", block_end_string="}",
    variable_start_string="\VAR{", variable_end_string="}",
 	comment_start_string = '\#{', comment_end_string = '#}',
 	line_comment_prefix = '%#',
    trim_blocks=True, lstrip_blocks=True
)

# import jinja2, os
# env = Environment(
# 	block_start_string = '\BLOCK{',
# 	block_end_string = '}',
# 	variable_start_string = '\VAR{',
# 	variable_end_string = '}',
# 	comment_start_string = '\#{',
# 	comment_end_string = '}',
# 	line_statement_prefix = '%%',
# 	line_comment_prefix = '%#',
# 	trim_blocks = True,
# 	autoescape = False,
# 	loader = jinja2.FileSystemLoader(os.path.abspath('.'))
# )

# striplen = 0.8
ticklen = 0.05
pad = 0.2
elong_proj = 0.1
# mulbl = r"$\gridmu{\theta}$"
# ulbl = r"$\gridu{t}$"
# mulbl = r"$\Rs_{\theta(t)}$ applied to $\gridmu{\theta}$"
mulbl = r"$\mv^1_t$ applied to $\gridmu{\theta}$"
ulbl = r"$\Rs_{\theta}$ applied to $\gridu{t}$"

color_fill = "Set1-B"
color_rad = "Set1-A"
color_proj = "Set1-C"
# color_fill = "Dark2-C"
# color_rad = "Dark2-B"
# color_proj = "Dark2-A"

def gen_one_grid(data, stripleft, stripright, projlbl, caption, radopts):
    proj_dir = np.array(data["proj_dir"])
    perp = np.array([-proj_dir[1], proj_dir[0]])
    trmat = np.array(data["trmat"])
    troffs = np.array(data["troffs"])
    grid_size = data["grid_size"]
    rad_range = data["rad_range"]
    start = (rad_range[0] - elong_proj) * proj_dir
    end = (rad_range[1] + elong_proj) * proj_dir

    def trafo(x, y):
        return troffs + trmat @ (x, y)

    mid = trafo(0.5, 0.5)
    corners = [trafo(x, y) for x, y in itertools.product(*2*[[0,1]])]
    xvals = [x for x, _ in corners]
    yvals = [y for _, y in corners]
    xmin, xmax = min(xvals), max(xvals)
    ymin, ymax = min(yvals), max(yvals)
    xmin -= pad
    ymin -= pad
    xmax += pad
    ymax += pad

    return env.from_string(r"""
%#\begin{tikzpicture}[baseline]%scale=5]
\begin{axis}[
%    xmin=\VAR{xmin}, xmax=\VAR{xmax},
%    ymin=\VAR{ymin}, ymax=\VAR{ymax},
    % axis line style={draw=none},
    % ticks=none,
    hide axis,
    unit vector ratio=1,
    anchor=midnode.center,
    clip=false
]

\node (midnode) at (\VAR{iter_to_coords(mid)}) {};

\BLOCK{for cell in data["cells"]}
\fill[\VAR{color_fill},opacity=\VAR{cell["inters"] | round(3)}] \VAR{iter_to_path(cell["corners"])};
%#\fill[blue,opacity=\VAR{cell["inters"] | round(3)}] \VAR{iter_to_path(cell["corners"], cs="axis cs")};
\BLOCK{endfor}

% want grid drawn over fills
%#\pgfplotsextra{
%#\draw[ultra thin, cm={\VAR{iter_to_coords(trmat.transpose().flatten())}, (\VAR{iter_to_coords(data["troffs"])})}] (0,0) grid [step=\VAR{1 / grid_size}] (1,1);
%#\draw[ultra thin] (0,0) grid [step=\VAR{1 / grid_size}] (1,1);
%#}

%#\begin{scope}[cm={\VAR{iter_to_coords(trmat.transpose().flatten())}, (\VAR{iter_to_coords(data["troffs"])})}]
\BLOCK{for i in range(grid_size + 1)}
%#\draw (\VAR{iter_to_coords(trafo(i / grid_size, 0))}) -- (\VAR{iter_to_coords(trafo(i / grid_size, 1))});
%#\draw (\VAR{iter_to_coords(trafo(0, i / grid_size))}) -- (\VAR{iter_to_coords(trafo(1, i / grid_size))});
\addplot[black] coordinates {(\VAR{iter_to_coords(trafo(i / grid_size, 0))}) (\VAR{iter_to_coords(trafo(i / grid_size, 1))})};
\addplot[black] coordinates {(\VAR{iter_to_coords(trafo(0, i / grid_size))}) (\VAR{iter_to_coords(trafo(1, i / grid_size))})};
%#\draw [use as bounding box] (\VAR{iter_to_coords(trafo(i / grid_size, 0))}) -- (\VAR{iter_to_coords(trafo(i / grid_size, 1))});
%#\draw [use as bounding box] (\VAR{iter_to_coords(trafo(0, i / grid_size))}) -- (\VAR{iter_to_coords(trafo(1, i / grid_size))});
\BLOCK{endfor}
%#\draw [red,cm={\VAR{iter_to_coords(trmat.flatten())}, (\VAR{iter_to_coords(troffs)})}] (0,0) rectangle (1,1);
%#\end{scope}
%#\useasboundingbox;

%#\draw[Set1-C,-Stealth] (\VAR{iter_to_coords(start)}) -- (\VAR{iter_to_coords(end)});
\addplot[\VAR{color_proj},-Stealth] coordinates {
    (\VAR{iter_to_coords(start)}) (\VAR{iter_to_coords(end)})
} node [below right] {\VAR{projlbl}};

%#\coordinate (ref) at (\VAR{iter_to_coords(5 * proj_dir)});
\BLOCK{for rad in data["allrads"]}
%#\coordinate (pt) at (\VAR{iter_to_coords(rad * proj_dir)});
%#\draw ($(pt)!0.1cm!90:(ref)$) -- ($(pt)!0.1cm!270:(ref)$);
\draw[\VAR{color_rad}](\VAR{iter_to_coords(rad * proj_dir + ticklen * perp)}) -- (\VAR{iter_to_coords(rad * proj_dir - ticklen * perp)});
\BLOCK{endfor}

\addplot[\VAR{color_rad}] coordinates {
    (\VAR{iter_to_coords(data["rads"][0] * proj_dir + stripleft * perp)}) (\VAR{iter_to_coords(data["rads"][0] * proj_dir - stripright * perp)})
} node [\VAR{radopts}] {$r_i$};
\addplot[\VAR{color_rad}] coordinates {
    (\VAR{iter_to_coords(data["rads"][1] * proj_dir + stripleft * perp)}) (\VAR{iter_to_coords(data["rads"][1] * proj_dir - stripright * perp)})
} node [\VAR{radopts}] {$r_{i+1}$};


%#\BLOCK{for rad in data["rads"]}
%#\addplot[Set1-A] coordinates {
    %#(\VAR{iter_to_coords(rad * proj_dir + stripleft * perp)}) (\VAR{iter_to_coords(rad * proj_dir - stripright * perp)})
%#};

%#\draw[green](\VAR{iter_to_coords(rad * proj_dir + striplen * perp)}) -- (\VAR{iter_to_coords(rad * proj_dir - striplen * perp)});
%#\coordinate (pt) at (\VAR{iter_to_coords(rad * proj_dir)});
%#\BLOCK{set perp = np.array(-proj_dir[1], proj_dir[0])}
%#\addplot coordinates {
    %#(\VAR{iter_to_coords(rad * proj_dir + 1 * perp)})
    %#(\VAR{iter_to_coords(rad * proj_dir - 1 * perp)})
%#};
%#draw[green] ($(pt)!\VAR{striplen}cm!90:(ref)$) -- ($(pt)!\VAR{striplen}cm!270:(ref)$);
%#\BLOCK{endfor}

%#\node [below=\belowcaptionskip of midnode] {\VAR{caption}};
\end{axis}
%#\end{tikzpicture}
    """).render(**dict(globals(), **locals())).strip()

# mu_plot = gen_one_grid(plot_data["mu"], 0.75, 0.75, r"$\theta(t)$", mulbl, #"below")
mu_plot = gen_one_grid(plot_data["mu"], 0.75, 0.75, r"$(1, t)$", mulbl, #"below")
    "pos=1,anchor=north")
u_plot = gen_one_grid(plot_data["u"], 0.5, 0.85, r"$\theta$", ulbl, "pos=1,anchor=north west")

# no subcaptions
# full = env.from_string(r"""
# \begin{tabular}{c@{\hskip 2cm}c}
# \VAR{mu_plot}
# &
# \VAR{u_plot}
# \end{tabular}
# """)

# captions aligned, but grids not horizontally aligned at center
# full = env.from_string(r"""
# \subcaptionbox{$\gridmu{\theta}$}[0.4\linewidth]{
# \VAR{mu_plot}
# }
# \subcaptionbox{$\gridu{t}$}[0.5\linewidth]{
# \VAR{u_plot}
# }
# """)

# https://tex.stackexchange.com/a/331133
# grids aligned horizontally, captions aligned horizontally, but captions not
# centered vertically under grids
# full = env.from_string(r"""
# %#\begin{tabular}{c@{\hskip 0.5cm}c}
# \begin{tabular}{cc}
# \VAR{mu_plot}
# &
# \VAR{u_plot}
# \\
# asdf
# %#\begin{minipage}{.4\linewidth}
# %#\leavevmode\subcaption{$\gridmu{\theta}$}
# %#\end{minipage}
# &
# ssadf
# %#\begin{minipage}{.4\linewidth}
# %#\leavevmode\subcaption{$\gridu{t}$}
# %#\end{minipage}
# \end{tabular}
# """)

full = env.from_string(r"""
\tikzsetnextfilename{grid}
\begin{tikzpicture}
\matrix[column sep=2cm,row sep=\abovecaptionskip]{
\VAR{mu_plot}
&
\VAR{u_plot}
\\
\node[inner sep=0pt]{(a) \VAR{mulbl}};
&
\node[inner sep=0pt]{(b) \VAR{ulbl}};
\\
}; % matrix
\end{tikzpicture}
""")

with open("./grid.tex", "w") as f:
    f.write(full.render(**locals()).strip())