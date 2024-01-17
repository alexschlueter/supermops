pycmd = export PYTHONPATH=..:$$PYTHONPATH; python

build/main.pdf: main.tex grid.tex
	mkdir -p figures build/figures
	latexmk -lualatex -shell-escape -synctex=1 -interaction=nonstopmode -file-line-error -outdir=build main

grid.tex: write_tex.py plot_data.json
	$(pycmd) write_tex.py

plot_data.json: | gen_plot_data.py ../loader.py
	$(pycmd) gen_plot_data.py

clean:
	rm -rf build figures grid.tex plot_data.json
