LATEXMK ?= latexmk
MAIN ?= main.tex
OUTDIR ?= build

.PHONY: all pdf clean distclean watch

all: pdf

pdf:
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=$(OUTDIR) $(MAIN)

watch:
	$(LATEXMK) -pdf -pvc -interaction=nonstopmode -halt-on-error -file-line-error -outdir=$(OUTDIR) $(MAIN)

clean:
	$(LATEXMK) -c -outdir=$(OUTDIR) $(MAIN)

distclean:
	$(LATEXMK) -C -outdir=$(OUTDIR) $(MAIN)
	$(RM) -r $(OUTDIR)
