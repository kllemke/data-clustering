# Review der Wissensbasis (Ticket #8)

Stand: 2026-11-16 · 19 Regeln · 0 Fehler · 19 Warnungen · Testfälle 9/9 bestanden

## Gefundene Probleme

| Schwere | Regel | Prüfung | Befund |
| --- | --- | --- | --- |
| Warnung | A01 | Quelle | Abschnitt im Kurs-Paper noch nicht eingetragen. |
| Warnung | A02 | Quelle | Abschnitt im Kurs-Paper noch nicht eingetragen. |
| Warnung | A03 | Quelle | Abschnitt im Kurs-Paper noch nicht eingetragen. |
| Warnung | R01 | Quelle | Abschnitt im Kurs-Paper noch nicht eingetragen. |
| Warnung | R02 | Quelle | Abschnitt im Kurs-Paper noch nicht eingetragen. |
| Warnung | R03 | Quelle | Abschnitt im Kurs-Paper noch nicht eingetragen. |
| Warnung | R04 | Quelle | Abschnitt im Kurs-Paper noch nicht eingetragen. |
| Warnung | R05 | Quelle | Abschnitt im Kurs-Paper noch nicht eingetragen. |
| Warnung | R06 | Quelle | Abschnitt im Kurs-Paper noch nicht eingetragen. |
| Warnung | R07 | Quelle | Abschnitt im Kurs-Paper noch nicht eingetragen. |
| Warnung | R08 | Quelle | Abschnitt im Kurs-Paper noch nicht eingetragen. |
| Warnung | R09 | Quelle | Abschnitt im Kurs-Paper noch nicht eingetragen. |
| Warnung | R10 | Quelle | Abschnitt im Kurs-Paper noch nicht eingetragen. |
| Warnung | R11 | Quelle | Abschnitt im Kurs-Paper noch nicht eingetragen. |
| Warnung | R12 | Quelle | Abschnitt im Kurs-Paper noch nicht eingetragen. |
| Warnung | R13 | Quelle | Abschnitt im Kurs-Paper noch nicht eingetragen. |
| Warnung | R14 | Quelle | Abschnitt im Kurs-Paper noch nicht eingetragen. |
| Warnung | R15 | Quelle | Abschnitt im Kurs-Paper noch nicht eingetragen. |
| Warnung | R16 | Quelle | Abschnitt im Kurs-Paper noch nicht eingetragen. |

## Testfälle

| Fall | Beschreibung | Ergebnis | Empfohlen | Gewarnt |
| --- | --- | --- | --- | --- |
| T01 | Iris-ähnlich: wenige numerische Merkmale, kaum Ausreißer | bestanden | Agglomerative (Ward), DBSCAN, GMM, HDBSCAN, K-Means, SOM | – |
| T02 | Umfrage mit rein kategorialen Antworten | bestanden | Agglomerative (Gower), K-Modes | K-Means |
| T03 | Kundendaten mit numerischen und kategorialen Spalten | bestanden | Agglomerative (Gower), K-Prototypes | K-Means |
| T04 | Numerische Daten mit vielen Ausreißern | bestanden | DBSCAN, HDBSCAN, SOM | K-Means |
| T05 | Zeitreihen, z. B. Stromverbrauch je Haushalt | bestanden | DBSCAN, SOM, TimeSeriesKMeans (DTW), k-Shape | – |
| T06 | Datenstrom, z. B. Sensordaten | bestanden | CluStream, DenStream | – |
| T07 | Sehr hochdimensional: DBSCAN muss gewarnt werden | bestanden | SOM | DBSCAN |
| T08 | Fallback: nur 5 Zeilen, keine gesicherte Empfehlung | bestanden | – | – |
| T09 | Fallback: unbekannter Datentyp | bestanden | – | – |

## Regeln im Überblick

| ID | Algorithmus | Urteil | Datentypen | Quelle |
| --- | --- | --- | --- | --- |
| R01 | K-Means | recommend | numeric | MacQueen (1967) |
| R02 | GMM | recommend | numeric | Dempster, Laird & Rubin (1977) |
| R03 | DBSCAN | recommend | numeric | Ester, Kriegel, Sander & Xu (1996) |
| R04 | HDBSCAN | recommend | numeric | Campello, Moulavi & Sander (2013) |
| R05 | Agglomerative (Ward) | recommend | numeric | Ward (1963) |
| R06 | K-Modes | recommend | categorical | Huang (1998) |
| R07 | K-Prototypes | recommend | mixed | Huang (1997) |
| R08 | Agglomerative (Gower) | recommend | mixed, categorical | Gower (1971) |
| R09 | TimeSeriesKMeans (DTW) | recommend | time_series | Petitjean, Ketterlin & Gançarski (2011) |
| R10 | k-Shape | recommend | time_series | Paparrizos & Gravano (2015) |
| R11 | DBSCAN | recommend | time_series | Ester, Kriegel, Sander & Xu (1996) |
| R12 | SOM | recommend | numeric, time_series | Kohonen (1990) |
| R13 | CluStream | recommend | stream | Aggarwal, Han, Wang & Yu (2003) |
| R14 | DenStream | recommend | stream | Cao, Ester, Qian & Zhou (2006) |
| R15 | Spherical K-Means | recommend | text | Dhillon & Modha (2001) |
| R16 | UK-Means | recommend | uncertain | Chau, Cheng, Kao & Ng (2006) |
| A01 | K-Means | avoid | categorical, mixed | Huang (1998) |
| A02 | K-Means | avoid | numeric | MacQueen (1967) |
| A03 | DBSCAN | avoid | numeric | Ester, Kriegel, Sander & Xu (1996) |
