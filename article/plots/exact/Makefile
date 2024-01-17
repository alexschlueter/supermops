pycmd = export PYTHONPATH=..:$$PYTHONPATH; python

build/main.pdf: main.tex exact.tex
	mkdir -p figures build/figures
	latexmk -pdf -shell-escape -synctex=1 -interaction=nonstopmode -file-line-error -outdir=build main

exact.tex: write_tex.py plot_data.json
	$(pycmd) write_tex.py

plot_data.json: | gen_plot_data.py ../loader.py
	$(pycmd) gen_plot_data.py

clean:
	rm -rf build figures exact.tex plot_data.json
