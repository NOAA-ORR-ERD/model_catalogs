## Pull Request Reminders

- [ ] Any tests that connect to model servers are marked with the `--runslow` decorator and won't be run by GitHub Actions. You should run these locally with `pytest --runslow`.
- [ ] Make sure the notebooks in the `docs` directory all run.
- [ ] Add tests for the new functionality.
- [ ] Add a bullet to `docs/whats_new.rst` describing your new work. If not already present, add a new section at the top of the document stating "[expected new version number] (unreleased)", for example: "v0.7.3 (unreleased)"
