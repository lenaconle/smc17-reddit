# Reddit Projekt SS17

Die Dateien `classifier_{rawtext,nolinks,nonumbers,lemma,lemmanumlink}.py` enthalten jeweils die Klassifikatoren, die auf den balancierten Politik-Kommentaren trainiert werden und zuvor der jeweiligen Vorverarbeitung unterliefen:
Diese Dateien greifen auf das Datenset `final_dataset.json` zu.

- rawtext : Rohdaten ohne Bereinigung
- nonumbers : Zahlen entfernt
- nolinks : URL durch Platzhalter ersetzt
- lemma : lemmatisiert
- lemmanumlink : alle drei Bereinigungsschritte

Der Algorithmus (SVM - Logistic Regression) lässt sich jeweils im Code austauschen.

Die Datei `classifier_nolinks_unb.py` enthält den Code für den Klassifikator, der wie in `classifier_nolinks.py` trainiert aber dann auf unbalancierten Politik-Daten getestet wurde.

Die Dateien `classifier_{hockey,thedonald}.py` enthalten die Tests auf den (unbalancierten) Hockey- bzw. TheDonald-Daten.
Diese Dateien greifen sowohl auf `final_dataset.json` als auch jeweils auf `hockey.json` oder `thedonald.json` zu.

Um zum Beispiel einen Klassifikator auf dem balancierten Politik-Datenset zu trainieren, nachdem die Wörter lemmatisiert wurden und das Modell dann auf den selben Daten zu testen, führt man aus: 
```
python classifier_lemma.py
```
