# Sunny Studio — demande naturelle vers PR

Cette modification est une preuve de pipeline : une demande écrite dans le cockpit devient une tâche, un workspace isolé et une proposition GitHub contrôlée.

## Étapes exécutées

1. La session d’orchestration a reçu la demande en langage naturel.
2. Le control-plane a créé une tâche et l’a synchronisée avec Vikunja.
3. Le worker de la tour a cloné le dépôt dans une branche `sunny-studio/...`.
4. Les dépendances et les vérifications disponibles ont été exécutées.
5. L’agent a travaillé avec un manifeste d’outils limité au workspace.
6. Le commit et la PR restent soumis à l’approbation explicite du cockpit.

## Contrôle

Le fichier `TOOLS.md` à la racine du workspace décrit les outils autorisés, les actions disponibles et la politique réseau. Aucun secret n’est copié dans le dépôt et l’agent ne pousse jamais directement.

