LATEXMK ?= latexmk
MAIN ?= main.tex
OUTDIR ?= build

.PHONY: all pdf link clean distclean watch

all: link

pdf:
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=$(OUTDIR) $(MAIN)

link: pdf
	ln -f $(OUTDIR)/main.pdf spritecraft-report.pdf

watch:
	$(LATEXMK) -pdf -pvc -interaction=nonstopmode -halt-on-error -file-line-error -outdir=$(OUTDIR) $(MAIN)

clean:
	$(LATEXMK) -c -outdir=$(OUTDIR) $(MAIN)

distclean:
	$(LATEXMK) -C -outdir=$(OUTDIR) $(MAIN)
	$(RM) -r $(OUTDIR)
