import json

from jinja2 import Environment
import numpy as np
import pandas as pd

with open("plot_data.json", "r") as file:
    plot_data = json.load(file)

combine_axlbls = True
transposed = False
pad = 1
blob_scale = 0.8
times = [3, 5, 7]

env = Environment(
    block_start_string="\BLOCK{", block_end_string="}",
    variable_start_string="\VAR{", variable_end_string="}",
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

blob_data = plot_data["blobs"]
allblobs = []
for t in times:
    gridy, gridx = np.mgrid[-pad:pad:2j, -pad:pad:2j]
    lbls = [f"fc2/t{t}gs100/mu0nu0", f"fc2/t{t}gs100/mu5nu3"]
    frame = pd.DataFrame.from_dict(blob_data[str(t)])
    sum_mat = frame.to_numpy()
    # print(gridx, gridy, frame, frame.to_numpy().flatten())
    blobs = zip(zip(gridx.flatten(), gridy.flatten(), sum_mat.flatten(), np.sqrt(blob_scale * sum_mat.flatten())),
        map(lambda x: [x], [0, 1, 0, 0]))
    allblobs.append((t, blobs))

lines_data = plot_data["lines"]
print(lines_data)
thetas = r"|\Theta|"
sigmas = r"|\Sigma|"
# line_legend = {
#     "mu0nu0": "static",
#     "adcg": "ADCG",
#     **{f"mu{mu}nu{nu}": f"${thetas}={mu}, {sigmas}={nu+t}$" for mu, nu in [(3, 0), (5, 3), (10, 7)]}
# }
# lines = [[(line_legend[parm], zip(xvals, line)) for parm, line in lines_data[str(t)].items()] for t in [3, 5, 7]]
def line_legend(t, parm):
    # if parm == "mu0nu0":
    #     return "static"
    # elif parm == "adcg":
    #     return "ADCG"
    # else:
    #     mu, nu = map(int, re.fullmatch(r"mu(\d+)nu(\d+)", parm).groups())
    #     return f"${thetas}={mu}, {sigmas}={nu+t}$"
    return {
        "mu0nu0": "static",
        "adcg": "ADCG",
        "mu3nu0": r"dimred.\ low",
        "mu5nu3": r"dimred.\ mid",
        "mu10nu7": r"dimred.\ high"
    }[parm]

lines = []
for t in times:
    bins = np.array(lines_data[str(t)]["bins"])
    print(bins)
    xvals = (bins[1:] + bins[:-1]) / 2
    lines_for_time = []
    for parm, line in lines_data[str(t)].items():
        if parm != "bins":
            lines_for_time.append((parm, zip(xvals, line)))
    lines.append((t, lines_for_time))
# lines = [(t, [(parm, zip(xvals, line)) for parm, line in lines_data[str(t)].items()]) for t in [3, 5, 7]]

blob_plots = [env.from_string(r"""
\tikzsetnextfilename{exact_blobs_t\VAR{t}}
\begin{tikzpicture}[baseline]
\begin{axis}[
    blob axis style,
    title={$|\mathcal{T}|=\VAR{t}$},
    \BLOCK{if transposed}
    title style={at={(0,0.5)},rotate=90},%,yshift=-0.1},
    \BLOCK{endif}
    \BLOCK{if not combine_axlbls or t == 3}
    ylabel={static},
    yticklabels={correct recons., failed recons.},
    ytick=data,
    % yticklabels={correct, failed},
    \BLOCK{endif}
]
    \addplot [blob plot style] table {
        x y count size style\\
        \BLOCK{for blobnums, blobopts in timeblobs}
        \VAR{blobnums | map("round", 2) | join(" ")} \VAR{blobopts | join(" ")}\\
        \BLOCK{endfor}
    };

\end{axis} 
\begin{axis}[blob cross style]
\end{axis}
\end{tikzpicture}%
""").render(**dict(globals(), **locals())).strip() for t, timeblobs in allblobs]

line_plots = [env.from_string(r"""
\tikzsetnextfilename{exact_lines_t\VAR{t}}
\begin{tikzpicture}[baseline]
\begin{axis}[
    lines axis style,
    \VAR{- "ylabel={Correct reconstruction rate}," if not combine_axlbls or t == 3 -}
    \VAR{- "ignore legend," if not t == 3 -}
]
\BLOCK{for parm, line in time_lines}
    \addplot+ [
        lines plot style,
    ] table {
        x y \\
        \BLOCK{for point in line}
        \VAR{point | map("round", 4) | join(" ")} \\
        \BLOCK{endfor}
    };
    \addlegendentry{\VAR{line_legend(t, parm)}}
\BLOCK{endfor}
\end{axis} 
\end{tikzpicture}%
""").render(**dict(globals(), **locals())).strip() for t, time_lines in lines]

plots = np.array([blob_plots, line_plots])
if transposed:
    plots = plots.transpose()

cross_len = 1.8

exact_styles = env.from_string(r"""
\pgfplotsset{
all axes style/.style={
    width=\linewidth/3,
    scale only axis,
    % ignore legend,
    every axis plot/.append style={
        thick
    }
},
blob axis style/.style={
    %axis equal image,
    height=\linewidth/3,
    title style={yshift=2.5em},
    all axes style,
    axis lines=box,
    axis line style={draw=none},
    xmin=-2, xmax=2,
    ymin=-2, ymax=2,
    % ticks=none,
    % xtick=data, ytick=data,
    xtick=data, ytick=\empty,
    xticklabels={correct recons., failed recons.},
    % xticklabels={O, O},
    % yticklabels={correct recons., failed recons.},
    xtick pos=top,
    % clip=false,
    ytick pos=left,
    tick align=inside,
    % xtick distance=2,
    % xticklabel style={
    %     shift={(0,0 |- {axis description cs:0,-1})}
    % },
    yticklabel style={
        % shift={(0,0 -| {axis description cs:-0.53,0})},
        rotate=90
    },
    xlabel={dynamic},
    % ylabel={static},
    label style={font=\bfseries},
    title style={font=\bfseries},
    y dir=reverse
    % axis line style={-}
},
blob plot style/.style={
    % mark=blob,
    mark=*,
    mark options={
        fill=YlGn-I,
        % draw=YlGn-K,
        % line width=4pt,
        %thick
    },
    scatter,
    % color={Set2-A},
    only marks,
    % scatter src=none,
    % point meta=explicit symbolic,
    % point meta=none,
    visualization depends on={value \thisrow{count} \as \blobcount},
    % visualization depends on={value \thisrow{anchor} \as \blobanchor},
    % visualization depends on={value \thisrow{color} \as \blobcolor},
    visualization depends on={value \thisrow{style} \as \blobstyle},
    visualization depends on={\thisrow{size} \as \marksize},
    scatter/@pre marker code/.style={
        /tikz/mark size=\marksize
    },
    scatter/@post marker code/.style={},
    %nodes near coords*={\pgfmathprintnumber\blobcount},
    nodes near coords*={\blobcount},
    every node near coord/.code={
            % color=\blobcolor
        \ifnum \blobstyle=0
            \tikzset{color=white}
        \else
            \tikzset{color=black}
        \fi
        },
    nodes near coords align={
        %anchor=\blobanchor,xshift=3pt
        \ifnum \blobstyle=0
            anchor=center,xshift=0pt
        \else
            anchor=west,xshift=3pt
        \fi
        },
},
blob cross style/.style={
    all axes style,
    height=\linewidth/3,
    axis lines=center,
    % xmin=\VAR{ -cross_len }, xmax=\VAR{cross_len},
    % ymin=\VAR{ -cross_len }, ymax=\VAR{cross_len},
    xmin=-1, xmax=1,
    ymin=-1, ymax=1,
    ticks=none,
    % ticks=both,
    axis line style={-}
},
lines axis style/.style={
    all axes style,
    height=6 cm,
    xlabel={Dynamic separation},
    % ylabel={Correct reconstruction rate},
    xticklabel style={
        /pgf/number format/fixed,
        /pgf/number format/precision=2
    },
    grid=major,
    xmin=0, xmax=0.1,
    ymin=-0.02, ymax=1.05,
    legend pos=north west,
    reverse legend,
},
lines plot style/.style={
    mark=none
},
}
""")

exact = env.from_string(r"""
\BLOCK{include exact_styles}

\centerline{%
%\begin{tabular}{c|c|c}
\begin{tabular}{\VAR{"c" * plots.shape[1]}}
\BLOCK{for plot_row in plots}
    \BLOCK{for plot in plot_row}
\VAR{plot}
\VAR{"&" if not loop.last}
    \BLOCK{endfor}
\VAR{"\\\\[0.5cm]" if not loop.last}
\BLOCK{endfor}
\end{tabular}%
}%
""")

with open("./exact.tex", "w") as f:
    f.write(exact.render(**locals()).strip())
