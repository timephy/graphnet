# Research to improve the network

Various ways of trying to improve network performance.
Some by changing some parts of the network design/architecture.
Some by including data about the ice.

## Overview

Name | Description | Relative Improvement | Implementation
--- | --- | --- | ---
main | graphnet main branch | 0% | default
main + edge self connections | node self connecting edges dont count towards total | ? | x
main + 3 NNB | graphnet main branch + graph with 3 nearest neighbors | ? | x
main + 4 NNB | graphnet main branch + graph with 4 nearest neighbors | ? | x
main + 5 NNB | graphnet main branch + graph with 5 nearest neighbors | ? | x
main + nodes carry ice data | graph nodes carry ice data as extra columns | ? | x
main + edge are connected by optical distance | graph edges are connected by X nnb by optical distance instead of actual distance | ? | x
main + edge are weighted by optical distance | graph edges attention (factor) is based on somehing like exp(-d^2) + max nnb or all | ? | x
main + attention learned by network | graph edges attention learned by network | ? | x


## To-Do

- [ ] Low Energy Selection
- [ ] Resolution plots add uncertainty
- [ ] Resolution plots save to file and overlay multiple plots
- [ ] Modify script to be flexible in modules/etc and set output dirs correctly
