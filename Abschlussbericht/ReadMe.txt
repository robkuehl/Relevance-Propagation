Die Hauptdatei, die die übersetzt werden muss, ist die Datei "layout.tex". Die Übersetzung kann mit PDFLatex erfolgen. Soll die Übersetzung auf andere Weise erfolgen, muss das Logo ggf. von jpg in ein anderes Format konvertiert werden.
Diese muss in Bezug auf Titel, Name etc angepasst werden.
Einleitung.tex ist eine separate Datei für die Einleitung, so kann sie korrekt ins Inhaltsverzeichnis übernommen werden, ebenso Erklaerung.tex, die auf jeden Fall unterschrieben werden muss!
In die Datei thesisbib.bib sollten die verwendeten (und zitierten!) Quellen eingetragen werden. Quellen sollte IMMER mit Seitenzahl oder Angabe des Theorems/Lemmas etc. zitiert werden, das heisst \cite[S. 127]{QUELLE} oder \cite[Theorem~4.19]{QUELLE}.
In die Datei main.tex kommt der Hauptteil der Arbeit. Es können auch für verschiedene Kapitel weitere Dateien definiert werden, sie müssen dann zwingend in layout.tex eingebunden werden, dies funktioniert analog zur main.tex.
Alternativ können die eingebundenen Dateien auch in die Hauptdatei kopiert werden, sodass die Arbeit dann nur aus einer .tex Datei besteht.

Das Format "book" wurde gewählt, damit die Ränder korrekt gesetzt werden und der Ausdruck später im Buchformat erfolgen kann. Zur besseren Lesbarkeit auf digitalen Geräten kann natürlich auch ein anderes Format wie "article" gewählt werden. Bitte beachten Sie dabei, dass sich das ganze Format noch einmal verschiebt, wenn kurz vor dem Druck dann wieder der Buchstil gewählt wird.

Bei Fragen:
https://en.wikibooks.org/wiki/LaTeX/
oder
martin.kuehn@uni-koeln.de
