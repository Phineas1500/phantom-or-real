# 27B Big-L0 Feature Mini-Dashboard

Created: `2026-04-30T02:57:12.678701+00:00`

Activations are measured at the cached last pre-generation prompt position.

## Summary

Rank comes from the trained sparse probe coefficient report; AUC is the univariate score from this feature alone.

| Feature | Property rank | Property density | Property AUC | Subtype rank | Subtype density | Subtype AUC | Notes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 72374 | 24 | 0.258 | 0.641 | 5 | 0.153 | 0.759 |  |
| 35036 | 20 | 0.189 | 0.400 | 16 | 0.392 | 0.343 |  |
| 4892 | 17 | 0.273 | 0.524 | 28 | 0.156 | 0.488 |  |
| 75345 | 4 | 0.227 | 0.548 |  | 0.299 | 0.516 |  |
| 187589 | 93 | 0.097 | 0.574 | 6 | 0.156 | 0.754 |  |
| 45599 | 77 | 0.048 | 0.488 | 15 | 0.012 | 0.497 |  |

## Top Activating Examples

### Feature 72374

- LLM short name: height-1 direct generalization pattern
- LLM relevance: likely_surface_or_style
- LLM steering priority: low
- LLM hypothesis: This feature appears to activate on very simple one-step ontology prompts where several named individuals share the same source class and the same observed property or subtype, making a direct universal hypothesis salient; its apparent correctness association is strongly entangled with task height and prompt simplicity.

#### `infer_property`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 424.0 | 1 | True | None | Vumpuses are not happy | Vumpuses are not happy.\n |
| 364.0 | 1 | True | None | Each vumpus is not aggressive | Vumpuses are not aggressive.\nAggression is not a property of vumpuses.\nThe observed vumpuses are a non-aggressive subset of vumpuses.\nThere is a factor pr... |
| 364.0 | 1 | True | None | Each yumpus is amenable | Yumpuses are amenable.\n |
| 364.0 | 1 | True | None | Dulpins are opaque | Dulpins are opaque. |
| 352.0 | 1 | True | None | Wumpuses are angry | Wumpuses are angry.\nWumpuses are not angry.\nAnger is not related to being a wumpus.\nBeing a wumpus causes anger.\nThere is a third factor causing both wum... |

#### `infer_subtype`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 356.0 | 1 | True | None | Each impus is a hilper | All impus are hilpers.\n |
| 332.0 | 1 | True | None | Each impus is a thorpin | All impus are thorpins. |
| 328.0 | 1 | True | None | Yumpuses are rimpees | All yumpuses are rimpees.\nBeing a yumpus does not guarantee being a rimpee. |
| 320.0 | 1 | True | None | Each impus is a serpee | All impus are serpees.\n |
| 312.0 | 1 | True | None | Each zumpus is a dropant | All zumpuses are dropants.\n |

### Feature 35036

- LLM short name: complex common-supertype / reversed hypothesis risk
- LLM relevance: maybe_reasoning_related
- LLM steering priority: medium
- LLM hypothesis: This feature seems to activate on harder ontology prompts where several observed individuals are connected by multi-hop subtype chains to a shared target concept, especially cases in which the model later gives reversed-direction or over-enumerated hypotheses rather than the single target-level generalization.

#### `infer_property`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 424.0 | 4 | False | wrong_direction | Kurpors are nervous | Being nervous is a property of irpers.\nBeing nervous is a property of kurpors.\nBeing nervous is a property of gwompants.\nBeing nervous is a property of pr... |
| 420.0 | 3 | False | wrong_direction | Daumpins are moderate | Moderate things are stirpors.\nModerate things are not stirpors.\nModerate things are daumpins.\nModerate things are not daumpins.\nModerate things are parpe... |
| 412.0 | 4 | False | wrong_direction | Shalpists are fast | Fast things are shalpists.\nFast things are vumpuses.\nFast things are quimpants.\nFast things are tergits.\nFast things are numpuses.\n |
| 400.0 | 3 | False | unnecessary | Every orgit is cold | Being cold is a property of folpees.\nBeing cold is a property of orgits.\nBeing cold is a property of kergits.\nBeing cold is a property of harpins.\nBeing ... |
| 396.0 | 3 | False | wrong_direction | Every stirpor is moderate | Moderate things are stirpors.\nModerate things are storpists.\nModerate things are borpins.\nModerate things are fomples.\nModerate things are phorpists. |

#### `infer_subtype`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 428.0 | 4 | False | wrong_direction | Dumpuses are dolpers | Dolpers are dumpuses.\nDolpers are not dumpuses. |
| 388.0 | 4 | False | wrong_direction | Yumpuses are frompors | Frompor is a hilper.\nFrompor is a porpor.\nFrompor is a yumpus.\nFrompor is a rorpant.\nFrompor is a yempors.\nFrompor is not a shumple.\nFrompor is not a t... |
| 376.0 | 3 | False | wrong_direction | Every fimple is a remple | A remple is a fimple.\nA werpor is a fimple.\nA rompus is a remple.\n |
| 372.0 | 4 | False | unnecessary | Every zhomple is a wumpus | Each yumpus is a wumpus.\nEach zhomple is a wumpus.\nEach lemper is a wumpus.\nEach storpist is a wumpus.\nEach rompus is a wumpus. |
| 364.0 | 3 | False | trivial | Grimpants are yempors | Raymond is a wumpus.\nNicole is a sarper.\nAmanda is a wumpus.\n |

### Feature 4892

- LLM short name: common-superclass hypothesis candidate
- LLM relevance: maybe_reasoning_related
- LLM steering priority: medium
- LLM hypothesis: This feature seems to activate on prompts where the observations can be explained by assigning the observed predicate to a shared target superclass/common category reached by several individuals through ontology chains, especially in non-height-1 cases.

#### `infer_property`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 400.0 | 4 | True | None | Each gergit is not translucent | Gergits are not translucent.\nJempors are not translucent.\nArpers are not translucent.\nDelpees are not translucent.\nYempors are not translucent.\nThorpins... |
| 400.0 | 4 | True | None | Grimpants are not happy | Grimpants are not happy.\nStergits are not happy.\nLirpins are not happy.\nKergits are not happy.\nUrpants are not happy.\nPergits are not happy.\nZhorpants ... |
| 360.0 | 3 | True | None | Every yimple is not happy | Yimples are not happy.\nUmpists are not happy.\nFolpees are not happy.\nKergits are not happy.\nBongits are not happy.\nLirpins are not happy.\nJompuses are ... |
| 344.0 | 2 | True | None | Jempors are not angry | Jempors are not angry. |
| 328.0 | 3 | True | None | Each yompin is not bright | Yompins are not bright.\nRemples are not bright.\nBongits are not bright.\nArpers are not bright.\nWerpors are not bright.\nLempers are not bright.\nYimples ... |

#### `infer_subtype`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 260.0 | 4 | False | wrong_direction | Numpuses are yerpists | Yerpists are numpuses.\nYerpists are not starples.\nYerpists are not stopins.\nYerpists are not zumpuses.\nYerpists are not dalpists.\nYerpists are not tergi... |
| 244.0 | 2 | True | None | Kurpors are porpors | Each kurpor is a porpor. |
| 236.0 | 3 | False | wrong_direction | Each zilpor is a hilper | Hilpers are zilpors.\nHilpers are not zilpors.\nZilpors are hilpers.\nZilpors are not hilpers.\n |
| 236.0 | 4 | False | unnecessary | Every fimple is an umpist | Laura is an umpist.\nKathleen is an umpist.\nSandra is an umpist.\nEach umpist is a yerpist.\nEach umpist is a hilper.\nEach umpist is a lompee.\nEach umpist... |
| 232.0 | 4 | False | wrong_direction | Every wolpee is a rorpant | Rorpant is wolpee.\nRorpant is frompor.\nRorpant is jompus.\nRorpant is umpist.\nRorpant is gwompant.\nRorpant is lerpant.\n |

### Feature 75345

- LLM short name: moderate-depth common superclass
- LLM relevance: maybe_reasoning_related
- LLM steering priority: medium
- LLM hypothesis: This feature seems to activate on ontology prompts where the observed entities can be unified by a shared target concept, especially in moderate-depth subclass chains, but its relationship to correct reasoning is weak and mixed.

#### `infer_property`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 288.0 | 2 | False | unnecessary | Each parper is fast | Being a parper is being fast.\nBeing a yompin is being fast.\nBeing a lerpant is being fast.\nBeing a shumple is being fast. |
| 288.0 | 4 | False | wrong_direction | Borpins are discordant | Discordance is not being a pergit.\nDiscordance is not being a zhorpant.\nDiscordance is not being a remple.\nDiscordance is not being a stirpor.\nDiscordanc... |
| 286.0 | 3 | True | None | Parpers are feisty | Parpers are feisty.\nLirpins are feisty.\nHarpins are feisty.\nZilpors are feisty.\nQuimpants are feisty.\nStopins are feisty.\nLompees are feisty. |
| 284.0 | 3 | False | wrong_direction | Each kurpor is discordant | Discordant things are not kurpors.\nDiscordant things are not daumpins.\nKurpors are not discordant.\nDaumpins are not discordant.\nShimpees are discordant.\... |
| 276.0 | 2 | False | unnecessary | Gorpees are salty | Being a gorpee is being salty.\nBeing a harpin is being salty.\nBeing a yempor is being salty.\nAll people are salty. |

#### `infer_subtype`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 330.0 | 3 | False | wrong_direction | Shalpists are quimpants | Being a shalpist is being a quimpant.\nBeing a quimpant is not being a shalpist. |
| 302.0 | 3 | True | None | Every remple is a shimpee | Remples are shimpees.\nFelpers are shimpees.\nZhorpants are shimpees.\nStirpors are shimpees.\nStarples are shimpees.\nGorpees are shimpees.\nJelgits are shi... |
| 298.0 | 2 | False | unnecessary | Each gomper is a shumple | Dalpists are shumples.\nStopins are shumples.\nShilpants are shumples.\nGompers are shumples. |
| 294.0 | 2 | True | None | Every serpee is a vumpus | Serpees are vumpuses.\nFomples are not vumpuses.\nWorples are not vumpuses.\nBeing a serpee is sufficient for being a vumpus. |
| 280.0 | 2 | False | wrong_direction | Every harpin is a rifpist | Rifpists are harpins.\nRifpists are lerpants.\nBeing a harpin is sufficient to be a rifpist.\nBeing a lerpant is sufficient to be a rifpist. |

### Feature 187589

- LLM short name: simple universal generalization
- LLM relevance: maybe_reasoning_related
- LLM steering priority: medium
- LLM hypothesis: This feature seems to activate on short, low-height prompts where several named individuals share the same source concept and the same observed predicate or observed type, making a direct universal hypothesis such as "all X are Y" or "all X have property P" salient.

#### `infer_property`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 278.0 | 2 | True | None | Each welgit is not transparent | Welgits are not transparent.\nUmpists are not transparent.\nKergits are not transparent.\n |
| 270.0 | 1 | True | None | Each remple is orange | Remples are orange.\n |
| 260.0 | 1 | True | None | Each harpin is blue | Harpins are blue.\nAll harpins are blue.\nBeing a harpin implies being blue.\n |
| 246.0 | 1 | True | None | Each tergit is blue | Tergits are blue.\nAll blue things are tergit.\nBeing a tergit causes blueness.\n |
| 242.0 | 2 | True | None | Every scrompist is blue | Scrompists are blue.\nHilpers are blue.\nIrpers are blue.\nBoompists are blue.\nBlue things are scrompists.\nBlue things are hilpers.\nBlue things are irpers... |

#### `infer_subtype`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 304.0 | 1 | True | None | Every kergit is a gergit | All kergits are gergits.\nNo kergits are gergits. |
| 298.0 | 1 | True | None | Kergits are vumpuses | All kergits are vumpuses. |
| 284.0 | 1 | True | None | Yumpuses are rimpees | All yumpuses are rimpees.\nBeing a yumpus does not guarantee being a rimpee. |
| 282.0 | 1 | True | None | Kergits are gergits | All kergits are gergits.\nNo kergits are gergits. |
| 278.0 | 1 | True | None | Each jempor is a remple | All jempors are remples.\n |

### Feature 45599

- LLM short name: shared-supertype fan-in with exhaustive hypothesis risk
- LLM relevance: maybe_reasoning_related
- LLM steering priority: low
- LLM hypothesis: This feature appears to activate on ontology prompts where several observed individuals are connected through different subtype chains to a common target concept, especially height-2/3 fan-in structures; top generations often enumerate multiple class-level hypotheses or reverse the intended property/type direction, but the evidence is not strong enough to treat it as a reliable correctness predictor.

#### `infer_property`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 394.0 | 3 | False | wrong_direction | Each gergit is not wooden | Wooden things are not gergits.\nWooden things are not orgits.\nWooden things are not tergits.\nWooden things are not rompus.\nWooden things are not ilpists. |
| 390.0 | 3 | False | wrong_direction | Every sorple is muffled | Being muffled is being a shilpant.\nBeing muffled is being a kergit.\nBeing muffled is being a remple.\nBeing muffled is being a sorple.\nBeing muffled is no... |
| 390.0 | 3 | False | unnecessary | Each zilpor is muffled | Being a zilpor is being muffled.\nBeing a rompus is being muffled.\nBeing a phorpist is being muffled.\nBeing a dolper is being muffled.\nBeing a yimple is b... |
| 374.0 | 3 | False | unnecessary | Each numpus is not muffled | Quimpants are not muffled.\nGergits are not muffled.\nBorpins are not muffled.\nTimples are not muffled.\nDolpers are not muffled.\nSerpees are not muffled.\... |
| 350.0 | 3 | True | None | Scrompists are not muffled | Scrompists are not muffled.\n |

#### `infer_subtype`

| Act. | H | Correct | Error | Ground truth | Output |
| ---: | ---: | --- | --- | --- | --- |
| 250.0 | 2 | False | unnecessary | Each shimpee is a bongit | Frompors are bongits.\nZumpuses are bongits.\nShimpees are bongits.\nTwimpees are bongits.\n |
| 238.0 | 2 | False | unnecessary | Every jelgit is a kurpor | Each kurpor is a jelgit.\nEach jelgit is a kurpor.\nBeing a kurpor is independent of being a drompee, jelgit, kergit, or umpist. |
| 230.0 | 3 | False | wrong_direction | Every yempor is a parper | Every parper is a yempor.\nEvery yempor is a parper.\nBeing a parper is equivalent to being a yempor. |
| 206.0 | 3 | False | unclassified | Each jelgit is a kurpor | Kurpor is jelgit.\nKurpor is numpus.\nKurpor is arper.\nKurpor is dumpus.\nKurpor is pergit.\nKurpor is boompist.\nBeing a kurpor is a property of being a je... |
| 198.0 | 3 | False | wrong_direction | Every impus is a bongit | Being a bongit is being a parper.\nBeing a bongit is being a shampor.\nBeing a bongit is being a rompus.\nBeing a bongit is being an impus.\nBeing a bongit i... |

