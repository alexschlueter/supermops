import json

from jinja2 import Environment
import numpy as np

with open("plot_data.json", "r") as file:
    plot_data = json.load(file)

env = Environment(
    block_start_string="\BLOCK{", block_end_string="}",
    variable_start_string="\VAR{", variable_end_string="}",
 	comment_start_string = '\#{', comment_end_string = '}',
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

def legend(parm):
    return {
        "mu5nu3_single": "dimred.\ $N=1$",
        "mu5nu3_double": "dimred.\ $N=2$",
        "mu5nu3_all": "dimred.\ $4\leq N\leq 20$",
        "adcg": "ADCG $4\leq N\leq 20$"
    }[parm]

noise_lvls = plot_data["noise_lvls"]
linesxy = [(legend(parm), zip(noise_lvls, wsteins))
    for parm, wsteins in plot_data["lines"].items()]

triang_ur = (1e3, 1e-3)
triang_ll_x = 1
triang_ll_y = triang_ur[1] * np.sqrt(triang_ll_x / triang_ur[0])

# ll_y und ur_x in rel axis cs
# ur_y in axis cs
# rel axis  convert to data

plot_tpl = env.from_string(r"""
\tikzsetnextfilename{noisy}
\begin{tikzpicture}
\begin{loglogaxis}[
    thick,
    legend pos=north west,
    width=10cm,
    xlabel={Noise level $\delta$},
    ylabel={Mean Wasserstein divergence},
    enlarge x limits=0.025,
    reverse legend,
]
\BLOCK{for name, line in linesxy}
    \addplot+ table {
        \BLOCK{for row in line}
        \VAR{row | map("round", 8) | join(" ")} \\
        \BLOCK{endfor}
    };
    \addlegendentry{\VAR{name}}
\BLOCK{endfor}

%\coordinate (ur) at (\VAR{triang_ur | join(", ")});

%\coordinate (ur) at ({rel axis cs:0.9,0}|-{axis cs:0,1e-3});
%\pgfplotspointgetcoordinates{(ur)}
% \def\urx{\pgfkeysvalueof{/data point/x}}
%\def\ury{\pgfkeysvalueof{/data point/y}}
%\pgfplotspointgetcoordinates{(rel axis cs:0,0.1)}
%\def\lly{\pgfkeysvalueof{/data point/y}}
%\node at (ur) {\urx \ury \lly};
%\node at (ur) {hi \urx a \pgfmathprintnumber{\pgfkeysvalueof{/data point/x}} b \pgfkeysvalueof{/pgfplots/xmin}};

\coordinate (ur) at ({rel axis cs:0.9,0}|-{axis cs:0,1e-3});
\pgfplotsextra{
\pgfplotspointgetcoordinates{(ur)}
\edef\urx{\pgfkeysvalueof{/data point/x}}
\edef\ury{\pgfkeysvalueof{/data point/y}}

\pgfplotspointgetcoordinates{(rel axis cs:0,0.1)}
\edef\lly{\pgfkeysvalueof{/data point/y}}

% \node at (ur) {hi \urx a $\pgfmathprintnumber{\pgfkeysvalueof{/data point/x}}$ b \pgfkeysvalueof{/pgfplots/xmin}};
\pgfmathsetmacro\llx{\urx*(\lly/\ury)^2}
}
% \node at (ur) {\llx \pgfmathprintnumber{\ury} \pgfmathprintnumber{\lly}};
\coordinate (ll) at (\llx,\lly);

%\coordinate (ll) at (\VAR{triang_ll_x}, \VAR{triang_ll_y});
\draw (ll) -| (ur) -- (ll) node[pos=0.5,sloped,anchor=north] {slope $\sqrt{\delta}$};
%\coordinate
%\pgfplotspointgetcoordinates{}

\end{loglogaxis}
\end{tikzpicture}
""")


with open("./noisy.tex", "w") as f:
    f.write(plot_tpl.render(**locals()).strip())
