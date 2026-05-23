# Qwen3.5 27B Qwen Scope W80K L0_100 Feature Mini-Dashboard

Created: `2026-05-23T19:41:14.159077+00:00`

Activations are measured at the cached last pre-generation prompt position.

## Summary

Rank comes from the trained sparse probe coefficient report; AUC is the univariate score from this feature alone.

| Feature | Property rank | Property density | Property AUC | Subtype rank | Subtype density | Subtype AUC | Notes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 7169 | 1 | 0.630 | 0.620 | 2 | 0.521 | 0.574 |  |
| 23296 | 2 | 0.465 | 0.479 | 5 | 0.333 | 0.568 |  |
| 22938 | 7 | 0.533 | 0.358 | 4 | 0.120 | 0.435 |  |
| 28984 | 8 | 0.951 | 0.542 | 8 | 0.942 | 0.627 | infer_property dense; infer_subtype dense |
| 68475 | 6 | 0.887 | 0.421 | 9 | 0.920 | 0.494 | infer_subtype dense |
| 4212 | 11 | 0.092 | 0.581 | 3 | 0.089 | 0.608 |  |
| 32398 | 12 | 0.997 | 0.587 | 6 | 0.933 | 0.702 | infer_property dense; infer_subtype dense |
| 51800 | 5 | 0.923 | 0.603 | 18 | 0.677 | 0.724 | infer_property dense |

## Top Activating Examples

### Feature 7169

#### `infer_property`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 5.1 | 2 | True | None | Umpists are not muffled | Umpists are not muffled. |
| 5.0 | 3 | True | None | Zhorpants are not sad | Zhorpants are not sad.\nWumpuses are not sad.\nBongits are not sad. |
| 4.9 | 2 | True | None | Dumpuses are not amenable | Dumpuses are not amenable. |
| 4.9 | 3 | True | None | Every wumpus is not fast | Wumpuses are not fast. |
| 4.9 | 2 | True | None | Dropants are earthy | Dropants are earthy. |

#### `infer_subtype`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 5.2 | 2 | True | None | Every orgit is a twimpee | All orgits are twimpees. |
| 4.7 | 1 | True | None | Dumpuses are bongits | A dumpus is a bongit. |
| 4.7 | 1 | True | None | Each dumpus is a shumple | A dumpus is a shumple. |
| 4.7 | 1 | True | None | Bempins are daumpins | A bempin is a daumpin. |
| 4.7 | 2 | True | None | Each gwompant is an urpant | Every gwompant is an urpant. |

### Feature 23296

#### `infer_property`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 12.8 | 1 | True | None | Every jompus is discordant | A jompus is discordant. |
| 12.5 | 1 | True | None | Each zilpor is amenable | A zilpor is amenable. |
| 12.4 | 1 | True | None | Each zhorpant is discordant | A zhorpant is discordant. |
| 12.2 | 2 | False | None | Each yempor is opaque | Every irper is opaque.\nEvery rofpin is opaque.\nEvery yempor is opaque. |
| 12.2 | 1 | True | None | Serpees are dull | A serpee is dull. |

#### `infer_subtype`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 13.3 | 1 | True | None | Each wolpee is a dumpus | A wolpee is a dumpus. |
| 13.1 | 1 | True | None | Each twimpee is a scrompist | A twimpee is a scrompist. |
| 13.1 | 1 | True | None | Each phorpist is a shilpant | A phorpist is a shilpant. |
| 13.0 | 1 | True | None | Every wolpee is a zhorpant | A wolpee is a zhorpant. |
| 12.9 | 1 | True | None | Fomples are shumples | A fomple is a shumple. |

### Feature 22938

#### `infer_property`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 7.2 | 4 | False | None | Delpees are not metallic | Dalpists are not metallic.\nDelpees are not metallic. |
| 7.2 | 4 | False | None | Every prilpant is not blue | Tumpuses are not blue.\nRemples are not blue.\nPrilpants are not blue. |
| 6.9 | 3 | False | None | Sorples are not snowy | Felpers are not snowy.\nZhomple are not snowy.\nSorples are not snowy. |
| 6.9 | 4 | False | None | Each chorper is not salty | Dalpists are not salty.\nChorpers are not salty. |
| 6.8 | 4 | True | None | Every gwompant is not fruity | Gwompants are not fruity.\nGrimpants are not fruity.\nArpers are not fruity. |

#### `infer_subtype`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 5.8 | 2 | False | None | Shilpants are phorpists | Scrompists are phorpists.\nShilpants are phorpists. |
| 5.6 | 2 | False | None | Kergits are dalpists | Aipers are dalpists.\nKergits are dalpists. |
| 5.5 | 2 | False | None | Every rompus is a gergit | Scrompists are gergits.\nRompuses are gergits. |
| 5.3 | 4 | False | None | Zhimpors are quimpants | Numpuses are quimpants.\nZhimpor are quimpants. |
| 4.9 | 4 | False | None | Wumpuses are scrompists | Orgits are scrompists.\nWumpuses are scrompists. |

### Feature 28984

#### `infer_property`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 7.2 | 1 | True | None | Each wolpee is loud | A wolpee is loud. |
| 7.2 | 1 | True | None | Wolpees are large | A wolpee is large. |
| 6.9 | 2 | True | None | Zhorpants are not large | Each zhorpant is not large. |
| 6.8 | 1 | True | None | Wolpees are spicy | A wolpee is spicy. |
| 6.8 | 1 | True | None | Wolpees are brown | A wolpee is brown. |

#### `infer_subtype`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 6.9 | 2 | False | None | Yempors are fomples | Every thorpin is a fomple.\nEvery yempor is a fomple.\nEvery kurpor is a fomple. |
| 6.9 | 2 | True | None | Yompins are gwompants | Every yompin is a gwompant. |
| 6.8 | 2 | True | None | Dalpists are zilpors | Every dalpist is a zilpor. |
| 6.7 | 2 | True | None | Zhimpors are jompuses | Every zhimpor is a jompus. |
| 6.6 | 2 | True | None | Every jempor is a felper | All jempors are felpers.\nAll timples are felpers.\nAll wolpees are felpers. |

### Feature 68475

#### `infer_property`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 12.9 | 2 | True | None | Each worple is not metallic | Each worple is not metallic. |
| 12.7 | 2 | False | None | Each impus is not earthy | Every borpin is not earthy.\nEvery shalpist is not earthy.\nEvery impus is not earthy. |
| 12.6 | 2 | False | None | Every lirpin is cold | Every rofpin is cold.\nEvery lirpin is cold. |
| 12.5 | 2 | True | None | Dumpuses are temperate | Each dumpus is temperate. |
| 12.4 | 2 | False | None | Quimpants are pale | Each ilpist is pale.\nJohn is pale. |

#### `infer_subtype`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 13.9 | 2 | False | None | Each kurpor is a numpus | Each gorpee is a numpus.\nEach porpor is a numpus.\nEach kurpor is a numpus. |
| 13.6 | 2 | True | None | Each yimple is a yempor | Each yimple is a yempor. |
| 13.6 | 3 | False | None | Rorpants are phorpists | Each bempin is a phorpist.\nEach rorpant is a phorpist. |
| 13.5 | 2 | True | None | Each scrompist is a dalpist | Each scrompist is a dalpist. |
| 13.4 | 2 | False | None | Shampors are shimpees | Every numpus is a shimpee. |

### Feature 4212

#### `infer_property`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 5.4 | 1 | True | None | Every dropant is not metallic | A dropant is not metallic. |
| 5.2 | 1 | True | None | Each dropant is not cold | A dropant is not cold. |
| 5.2 | 1 | True | None | Rofpins are not wooden | A rofpin is not wooden. |
| 5.2 | 1 | True | None | Yompins are not pale | A yompin is not pale. |
| 5.1 | 1 | True | None | Each remple is not opaque | A remple is not opaque. |

#### `infer_subtype`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 4.7 | 1 | True | None | Every rompus is a kurpor | A rompus is a kurpor. |
| 4.7 | 1 | True | None | Each jompus is a pergit | A jompus is a pergit. |
| 4.7 | 1 | True | None | Frompors are bempins | A frompor is a bempin. |
| 4.6 | 1 | True | None | Every frompor is a felper | A frompor is a felper. |
| 4.6 | 1 | True | None | Jompuses are shimpees | A jompus is a shimpee. |

### Feature 32398

#### `infer_property`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 10.4 | 2 | False | None | Chorpers are not temperate | Every timple is not temperate.\nEvery chorper is not temperate. |
| 10.4 | 2 | False | None | Each wolpee is not dark | Each scrompist is not dark.\nEach storpist is not dark.\nEach wolpee is not dark. |
| 10.1 | 2 | True | None | Bongits are not transparent | Every bongit is not transparent. |
| 10.0 | 1 | True | None | Every tumpus is temperate | A tumpus is temperate. |
| 10.0 | 1 | True | None | Each yumpus is amenable | A yumpus is amenable. |

#### `infer_subtype`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 10.4 | 1 | True | None | Yumpuses are starples | A yumpus is a starple. |
| 10.4 | 1 | True | None | Yumpuses are rimpees | A yumpus is a rimpee. |
| 10.3 | 1 | True | None | Vumpuses are rorpants | A vumpus is a rorpant. |
| 10.2 | 1 | True | None | Thorpins are yumpuses | A thorpin is a yumpus. |
| 10.1 | 1 | True | None | Thorpins are quimpants | A thorpin is a quimpant. |

### Feature 51800

#### `infer_property`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 10.8 | 1 | True | None | Every irper is sad | An irper is sad. |
| 9.9 | 2 | True | None | Porpors are not happy | Every porpor is not happy. |
| 9.9 | 4 | True | None | Each rorpant is not happy | Every rorpant is not happy.\nEvery vumpus is not happy.\nEvery harpin is not happy. |
| 9.8 | 1 | True | None | Boompists are not happy | A boompist is not happy. |
| 9.7 | 1 | True | None | Vumpuses are not happy | A vumpus is not happy. |

#### `infer_subtype`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 8.0 | 1 | True | None | Each zumpus is a dropant | A zumpus is a dropant. |
| 7.5 | 1 | True | None | Scrompists are boompists | A scrompist is a boompist. |
| 7.5 | 1 | True | None | Vumpuses are rorpants | A vumpus is a rorpant. |
| 7.4 | 1 | True | None | Lempers are chorpers | A lemper is a chorper. |
| 7.3 | 1 | True | None | Every vumpus is a fomple | A vumpus is a fomple. |
