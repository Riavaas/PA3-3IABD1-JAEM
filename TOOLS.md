# Tools autorisés — Sunny Studio

Mission : github-cbd5083a

- Fichiers du workspace (filesystem) : file.read, file.write, file.search · garde-fou : workspace-only
- Shell contrôlé (shell) : command.run, process.inspect · garde-fou : no secrets, no destructive root commands
- Git local (git) : git.status, git.diff, git.branch, git.commit · garde-fou : no push from agent
- Tests / build (tests) : test.detect, test.run, build.run · garde-fou : logs retained
- GitHub contrôlé (github) : repo.read, branch.create, pr.read · garde-fou : push/merge approval required
- Vikunja via control-plane (vikunja) : project.read, task.create, task.update · garde-fou : token stays server-side

Politique : workspaceOnly=true ; réseau=read-only ; push=approval-only.
