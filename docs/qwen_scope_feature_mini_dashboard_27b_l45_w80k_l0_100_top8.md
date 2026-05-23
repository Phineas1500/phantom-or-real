# Qwen3.5 27B Qwen Scope W80K L0_100 Feature Mini-Dashboard

Created: `2026-05-23T18:29:18.137829+00:00`

Activations are measured at the cached last pre-generation prompt position.

## Summary

Rank comes from the trained sparse probe coefficient report; AUC is the univariate score from this feature alone.

| Feature | Property rank | Property density | Property AUC | Subtype rank | Subtype density | Subtype AUC | Notes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 28479 | 5 | 1.000 | 0.384 | 5 | 1.000 | 0.302 | infer_property dense; infer_subtype dense |
| 69171 | 8 | 0.906 | 0.410 | 6 | 0.907 | 0.331 | infer_property dense; infer_subtype dense |
| 5666 | 2 | 1.000 | 0.588 | 10 | 1.000 | 0.603 | infer_property dense; infer_subtype dense |
| 80956 | 13 | 0.932 | 0.392 | 11 | 0.964 | 0.332 | infer_property dense; infer_subtype dense |
| 68759 | 4 | 1.000 | 0.580 | 19 | 1.000 | 0.691 | infer_property dense; infer_subtype dense |
| 77473 |  | 0.096 | 0.574 | 1 | 0.059 | 0.572 |  |
| 68363 |  | 0.000 | 0.500 | 2 | 0.025 | 0.530 |  |
| 67802 | 10 | 0.061 | 0.554 |  | 0.101 | 0.610 |  |

## Top Activating Examples

### Feature 28479

#### `infer_property`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 2.7 | 4 | True | None | Urpants are transparent | Every urpant is transparent.\nEvery sorple is transparent.\nEvery yumpus is transparent. |
| 2.7 | 3 | False | None | Twimpees are brown | Storpists are brown.\nRemples are brown.\nTwimpees are brown. |
| 2.7 | 3 | False | None | Every quimpant is happy | Prilpants are happy.\nQuimpants are happy. |
| 2.7 | 3 | False | None | Every lemper is orange | Every rorpant is orange.\nEvery grimpant is orange.\nEvery lemper is orange. |
| 2.7 | 4 | False | None | Each dolper is orange | Every dalpist is orange.\nEvery wumpus is orange.\nEvery dolper is orange. |

#### `infer_subtype`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 2.9 | 4 | False | None | Hilpers are dumpuses | Dumpuses are hilpers.\nDumpuses are dolpers.\nDumpuses are irpers. |
| 2.9 | 4 | False | None | Harpins are shumples | Shumples are harpins.\nShumples are gompers.\nShumples are jompus.\nShumples are storpists.\nShumples are sarper.\nShumples are pergit.\nShumples are serpee.... |
| 2.8 | 4 | False | None | Each jelgit is a porpor | Umpists are porpors.\nJelgits are porpors. |
| 2.8 | 4 | False | None | Zilpors are yimples | Yimples are zilpors.\nYimples are jompuses.\nYimples are delpees. |
| 2.8 | 4 | True | None | Each daumpin is a dumpus | Every daumpin is a dumpus.\nEvery kurpor is a dumpus. |

### Feature 69171

#### `infer_property`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 2.2 | 2 | False | None | Each frompor is not transparent | Gompers are not transparent.\nFrompors are not transparent. |
| 2.2 | 3 | False | None | Every hilper is not melodic | Every impus is not melodic.\nEvery timple is not melodic.\nEvery hilper is not melodic. |
| 2.2 | 3 | False | None | Every hilper is not transparent | Lirpins are not transparent.\nHilpers are not transparent. |
| 2.2 | 4 | False | None | Each rofpin is not melodic | Stopins are not melodic.\nRofpins are not melodic.\nUmpists are not melodic. |
| 2.1 | 2 | True | None | Stergits are not temperate | Stergits are not temperate.\nNumpuses are not temperate.\nOrgits are not temperate. |

#### `infer_subtype`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 2.2 | 4 | False | None | Each fomple is a boompist | Porpors are boompists.\nFomple is a boompist. |
| 2.1 | 3 | False | None | Every werpor is a wumpus | Shalpists are wumpuses.\nWerpors are wumpuses. |
| 2.1 | 3 | False | None | Each thorpin is an irper | Grimpants are irpers.\nThorpins are irpers. |
| 2.1 | 4 | True | None | Each gomper is an irper | Every gomper is an irper.\nEvery zhomple is an irper. |
| 2.1 | 4 | True | None | Each drompee is a zumpus | Each drompee is a zumpus. |

### Feature 5666

#### `infer_property`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 2.5 | 2 | True | None | Each dumpus is not happy | Each dumpus is not happy. |
| 2.5 | 2 | True | None | Each stirpor is not kind | Each stirpor is not kind. |
| 2.5 | 2 | True | None | Remples are not happy | Every remple is not happy. |
| 2.4 | 2 | False | None | Each umpist is not moderate | Folpees are not moderate.\nScrompists are not moderate.\nUmpists are not moderate. |
| 2.4 | 2 | True | None | Each dalpist is not kind | Dalpists are not kind. |

#### `infer_subtype`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 2.4 | 2 | True | None | Each ilpist is a dolper | All ilpists are dolpers. |
| 2.4 | 2 | False | None | Worples are yimples | Gompers are yimples.\nWorples are yimples.\nBoompists are yimples. |
| 2.4 | 2 | True | None | Dolpers are bongits | Every dolper is a bongit. |
| 2.4 | 2 | True | None | Orgits are irpers | Every orgit is an irper. |
| 2.4 | 2 | True | None | Fomples are hilpers | Fomples are hilpers. |

### Feature 80956

#### `infer_property`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 2.3 | 3 | False | None | Frompors are not happy | Dulpins are not happy.\nFrompors are not happy.\nGergits are not happy.\nSerpees are not happy. |
| 2.3 | 4 | False | None | Every porpor is not happy | Remples are not happy.\nPorpors are not happy. |
| 2.3 | 4 | True | None | Irpers are not happy | Every irper is not happy.\nEvery hilper is not happy.\nEvery lemper is not happy. |
| 2.3 | 4 | True | None | Rifpists are cold | Rifpists are cold.\nVumpuses are cold.\nGorpees are cold.\nWumpuses are cold.\nGrimpants are cold. |
| 2.3 | 3 | True | None | Every impus is not happy | Every impus is not happy. |

#### `infer_subtype`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 2.5 | 4 | True | None | Kergits are umpists | Each kergit is an umpist.\nEach stirpor is an umpist.\nEach pergit is an umpist. |
| 2.5 | 3 | True | None | Sorples are dropants | Every sorple is a dropant.\nEvery urpant is a dropant. |
| 2.4 | 3 | True | None | Each jompus is a stirpor | Every jompus is a stirpor.\nEvery tumpus is a stirpor.\nEvery wolpee is a stirpor. |
| 2.4 | 4 | False | None | Zhimpors are thorpins | Thorpins are kergits.\nThorpins are urpants.\nThorpins are zhimpors. |
| 2.4 | 4 | True | None | Each dulpin is a rimpee | Every dulpin is a rimpee.\nEvery rifpist is a rimpee.\nEvery lirpin is a rimpee. |

### Feature 68759

#### `infer_property`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 3.6 | 1 | True | None | Every gomper is spicy | A gomper is spicy. |
| 3.6 | 1 | True | None | Each kurpor is spicy | A kurpor is spicy. |
| 3.6 | 1 | True | None | Every tergit is sweet | A tergit is sweet. |
| 3.6 | 1 | True | None | Each twimpee is sweet | A twimpee is sweet. |
| 3.6 | 1 | True | None | Each zhomple is fruity | A zhomple is fruity. |

#### `infer_subtype`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 3.6 | 1 | True | None | Each jelgit is a bongit | A jelgit is a bongit. |
| 3.6 | 1 | True | None | Every yempor is a bongit | A yempor is a bongit. |
| 3.6 | 1 | True | None | Gorpees are felpers | A gorpee is a felper. |
| 3.6 | 1 | True | None | Werpors are jompuses | A werpor is a jompus. |
| 3.6 | 1 | True | None | Kurpors are yempors | A kurpor is a yempor. |

### Feature 77473

#### `infer_property`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 2.1 | 1 | True | None | Zumpuses are not hot | A zumpus is not hot. |
| 2.0 | 1 | True | None | Quimpants are not fast | A quimpant is not fast. |
| 2.0 | 1 | True | None | Each fomple is not melodic | A fomple is not melodic. |
| 1.9 | 1 | True | None | Yerpists are not large | A yerpist is not large. |
| 1.9 | 1 | True | None | Each shalpist is not rainy | A shalpist is not rainy. |

#### `infer_subtype`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 1.8 | 1 | True | None | Dropants are starples | A dropant is a starple. |
| 1.6 | 1 | True | None | Each dropant is a rifpist | A dropant is a rifpist. |
| 1.6 | 1 | True | None | Every vumpus is a daumpin | A vumpus is a daumpin. |
| 1.6 | 1 | True | None | Rofpins are twimpees | A rofpin is a twimpee. |
| 1.6 | 1 | True | None | Each dumpus is a lompee | A dumpus is a lompee. |

### Feature 68363

#### `infer_property`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 0.0 | 1 | True | None | Phorpists are not angry | A phorpist is not angry. |
| 0.0 | 1 | True | None | Bongits are sad | A bongit is sad. |
| 0.0 | 1 | True | None | Every borpin is dark | A borpin is dark. |
| 0.0 | 1 | True | None | Each starple is not earthy | A starple is not earthy. |
| 0.0 | 1 | True | None | Sarpers are translucent | A sarper is translucent. |

#### `infer_subtype`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 1.4 | 1 | True | None | Each folpee is a felper | A folpee is a felper. |
| 1.4 | 1 | True | None | Shalpists are umpists | A shalpist is an umpist. |
| 1.4 | 1 | True | None | Every dolper is a hilper | A dolper is a hilper. |
| 1.4 | 1 | True | None | Delpees are rifpists | A delpee is a rifpist. |
| 1.4 | 1 | True | None | Every jelgit is a welgit | A jelgit is a welgit. |

### Feature 67802

#### `infer_property`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 2.2 | 1 | True | None | Each felper is windy | A felper is windy. |
| 2.1 | 1 | True | None | Each hilper is discordant | A hilper is discordant. |
| 2.1 | 1 | True | None | Felpers are melodic | A felper is melodic. |
| 2.1 | 1 | True | None | Dumpuses are muffled | A dumpus is muffled. |
| 2.0 | 1 | True | None | Each fimple is not discordant | A fimple is not discordant. |

#### `infer_subtype`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 3.0 | 1 | True | None | Each hilper is an irper | A hilper is an irper. |
| 3.0 | 1 | True | None | Felpers are hilpers | A felper is a hilper. |
| 3.0 | 1 | True | None | Each shalpist is a dalpist | A shalpist is a dalpist. |
| 2.9 | 1 | True | None | Each fimple is a yimple | A fimple is a yimple. |
| 2.8 | 1 | True | None | Storpists are dalpists | A storpist is a dalpist. |
