# suivi resultats RBF

trace des runs

## runs


| Run        | Mode               | gamma  | K   | seed | epochs             | lr   | acc train | acc test  | notes                                                                                 |
| ---------- | ------------------ | ------ | --- | ---- | ------------------ | ---- | --------- | --------- | ------------------------------------------------------------------------------------- |
| R0         | pinv               | 0.01   | 100 | 42   | —                  | —    | 0.576     | 0.545     | run de ref avant rosenblatt                                                           |
| R1         | rosenblatt         | 0.01   | 100 | 67   | 100 (pas converge) | 0.1  | 0.418     | 0.402     | catastrophe, il repond wet a tout. en fait gamma trop grand, les phi valent quasi 0   |
| R1b        | pinv               | 0.01   | 100 | 67   | —                  | —    | 0.541     | 0.525     | pinv avec les memes centres, bcp mieux, vexant                                        |
| R2         | rosenblatt         | 0.001  | 100 | 67   | 300 (pas converge) | 0.01 | 0.535     | 0.472     | mieux mais la courbe fait nimporte quoi, test entre 0.43 et 0.60 selon ou on s'arrete |
| R2-K200    | rosenblatt         | 0.001  | 200 | 67   | 300                | 0.01 | 0.614     | 0.608     | la ! on passe devant pinv                                                             |
| R3         | rosenblatt+pocket  | 0.001  | 200 | 67   | 300                | 0.01 | 0.668     | 0.591     | premier essai en **rgb**. dry et wet se melangent                                     |
| R3-g0.0001 | rosenblatt+pocket  | 0.0001 | 200 | 67   | 300                | 0.01 | 0.622     | 0.621     | record. en rgb faut baisser gamma (3x plus de dims)                                   |
| R4         | rosenblatt+pocket  | 0.0001 | 300 | 67   | 300                | 0.01 | 0.637     | 0.635     | rgb, confusion enfin equilibree                                                       |
| R4-K400    | rosenblatt+pocket  | 0.0001 | 400 | 67   | 300                | 0.01 | 0.634     | 0.645     | record. a 600 ca redescend donc on reste a 400                                        |
| R5-s42     | ros+pocket+shuffle | 0.0001 | 400 | 42   | 300                | 0.01 | 0.637     | **0.661** | meilleur score. par contre wet en souffrance (53/106)                                 |
| R5-s67     | ros+pocket+shuffle | 0.0001 | 400 | 67   | 300                | 0.01 | 0.643     | 0.635     | confusion la plus propre des trois                                                    |
| R5-s96     | ros+pocket+shuffle | 0.0001 | 400 | 96   | 300                | 0.01 | 0.642     | 0.648     | entre les deux                                                                        |


## resultat final (valide sur 3 seeds)

config : rgb/normalisee gamma=0.0001 K=400 rosenblatt + pocket + shuffle epochs=300 lr=0.01


| seed    | train     | test              |
| ------- | --------- | ----------------- |
| 42      | 0.637     | 0.661             |
| 67      | 0.643     | 0.635             |
| 96      | 0.642     | 0.648             |
| **moy** | **0.641** | **0.648 ± 0.013** |


bilan : ~0.65 en test quelle que soit la seed, donc c'est pas un coup de bol du kmeans. pour situer : hasard 0.33, lineaire 0.40, notre pinv du debut 0.545. train et test collent partout donc pas de par coeur. curieux : les seeds donnent presque la meme acc mais pas les memes confusions (la 42 adore dry, la 67 est equilibree), normal vu que les centres tombent pas au meme endroit.

on fige la partie RBF ici. le train depasse jamais ~0.65 donc le probleme c'est plus les hyperparametres, c'est les features (pixels bruts). ca tombe bien le MLP arrive.

le chemin en gros : 0.402 -> 0.472 (gamma corrige) -> 0.608 (K200) -> 0.621 (rgb) -> 0.645 (K400) -> 0.648 (pocket+shuffle, 3 seeds)

## sweeps R1 (rosenblatt, K=100, seed=67, epochs=100, lr=0.1)

### gamma (K=100)


| gamma     | train | test      |
| --------- | ----- | --------- |
| 0.0001    | 0.485 | 0.505     |
| **0.001** | 0.523 | **0.512** |
| 0.01      | 0.418 | 0.402     |
| 0.1       | 0.355 | 0.346     |
| 1.0       | 0.345 | 0.322     |


donc l'optimum est vers 0.001, 10x plus petit que ce qu'on utilisait avec pinv. explication trouvee apres coup : avec d=16384 les distances^2 sont enormes, a gamma 0.01 e^(-g*d^2) donne quasi 0 partout, du coup les corrections lr*phi font rien et il n'apprend que les biais -> classe majoritaire, ce qui colle avec le R1 tout wet

### K (gamma=0.01)


| K   | train | test  |
| --- | ----- | ----- |
| 10  | 0.388 | 0.369 |
| 30  | 0.407 | 0.372 |
| 60  | 0.403 | 0.372 |
| 100 | 0.418 | 0.402 |
| 200 | 0.46  | 0.392 |


fait au mauvais gamma donc pas trop exploitable, refait au R2

### les 6 variantes (gamma=0.01, K=100)


| variante                  | test  |
| ------------------------- | ----- |
| nb/normalisee             | 0.402 |
| rgb/normalisee            | 0.362 |
| contours/normalisee       | 0.359 |
| toutes les non_normalisee | 0.322 |


les non normalisees font toutes pile 0.322, il predit une seule classe. pixels 0-255 = distances gigantesques = phi a 0. bref sans normalisation le rbf marche juste pas, on arrete de les tester (commentees dans le notebook)

## sweeps R2 (rosenblatt, gamma=0.001, seed=67, epochs=300, lr=0.01)

### gamma (K=100)


| gamma     | train | test      |
| --------- | ----- | --------- |
| 0.0001    | 0.344 | 0.342     |
| **0.001** | 0.535 | **0.472** |
| 0.01      | 0.434 | 0.412     |
| 0.1       | 0.364 | 0.346     |
| 1.0       | 0.345 | 0.322     |


0.001 toujours devant. bizarre par contre : 0.0001 marchait a 0.505 au R1 (lr 0.1) et la il s'ecroule avec lr 0.01. donc gamma et lr jouent ensemble, a garder en tete

### K (gamma=0.001)


| K       | train | test      |
| ------- | ----- | --------- |
| 10      | 0.477 | 0.478     |
| 30      | 0.496 | 0.492     |
| 60      | 0.528 | 0.525     |
| 100     | 0.535 | 0.472     |
| **200** | 0.614 | **0.608** |


ca monte quasi tout du long et train reste colle au test -> on est en sous apprentissage, faut plus de centres

### variantes (gamma=0.001, K=100)


| variante            | test      |
| ------------------- | --------- |
| **rgb/normalisee**  | **0.545** |
| nb/normalisee       | 0.472     |
| contours/normalisee | 0.359     |


tiens, rgb passe devant nb maintenant. logique en vrai, la couleur aide pour mouille (sombre, reflets) vs neige (blanc)

### rosenblatt vs pinv (gamma=0.001, K=100, memes centres)


| mode       | train | test  |
| ---------- | ----- | ----- |
| rosenblatt | 0.535 | 0.472 |
| pinv       | 0.639 | 0.581 |


pinv gagne encore a K=100, mais notre rosenblatt K=200 fait 0.608 donc ca va. de toute facon on reste en rosenblatt (consigne), cellule commentee dans le notebook

## sweeps R4 (rosenblatt+pocket, rgb, epochs=300, lr=0.01)

### gamma (K=300)


| gamma      | train | test      |
| ---------- | ----- | --------- |
| **0.0001** | 0.637 | **0.635** |
| 0.001      | 0.747 | 0.608     |
| 0.01       | 0.597 | 0.528     |


0.0001 confirme pour rgb. et a 0.001 on voit bien le par coeur arriver (0.747 train vs 0.608 test), a garder pour le rapport

### K (gamma=0.0001)


| K       | train | test      |
| ------- | ----- | --------- |
| 60      | 0.602 | 0.625     |
| 100     | 0.613 | 0.618     |
| 200     | 0.622 | 0.621     |
| 300     | 0.637 | 0.635     |
| **400** | 0.634 | **0.645** |
| 600     | 0.633 | 0.628     |


la fameuse cloche du cours en vrai : ca monte jusqu'a 400 et ca redescend a 600. et le train bouge plus non plus, on a atteint la limite de ce que ces features permettent

## problemes qu'on a eu + solutions

- la courbe oscillait dans tous les sens et le score dependait de la derniere epoch (loterie). c'est parce que c'est pas separable dans l'espace des phi donc le perceptron ne converge jamais. solution : **pocket**, on garde les meilleurs poids croises pendant le train et on rend ceux la (choisis sur l'acc train hein, pas test, sinon c'est de la triche). ligne `pocket epoch <e> train <acc>` dans la sortie
- l'ordre des exemples etait toujours le meme a chaque epoch, ca fait des cycles de corrections qui se repetent. solution : **shuffle** fisher-yates a chaque epoch (avec rand() donc ca reste reproductible via la seed). teste sur un petit dataset pas separable : pocket train passe de 0.467 a 0.688

