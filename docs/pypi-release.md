# Publishing XSTARS on PyPI

XSTARS 1.3.1 is a packaging-only release. The statistics, figures, and Excel/WPS behavior are unchanged. `pip install xstars` installs the stats and plot engine. `pip install "xstars[excel]"` adds xlwings and ttkbootstrap. Most users should keep using the installers on [GitHub Releases](https://github.com/Frankkk1912/xstars/releases).

There is no `CHANGELOG` file in this repository. This page is the release note for 1.3.1 and the maintainer runbook.

1.3.1 只改打包，不改统计、作图或 Excel/WPS 行为。普通用户继续用 GitHub Releases 里的安装包。本仓库没有 `CHANGELOG`，本页就是 1.3.1 的发布说明。

## One-time setup

Do this once, before the first upload. Do not create a PyPI API token.

### 1. Accounts, both with 2FA

- PyPI: https://pypi.org/account/register/ — account that will own the `xstars` project. Turn on two-factor authentication. PyPI requires 2FA to publish.
- TestPyPI: https://test.pypi.org/account/register/ — a separate account. TestPyPI does not share logins with PyPI. Turn on 2FA there too.

### 2. Pending trusted publishers

Create these **before** the project exists on the index, so the GitHub workflow claims the name `xstars`. The workflow file is `.github/workflows/publish.yml`. It publishes with GitHub OIDC (`id-token: write`). It does not read an API token.

On PyPI: Account → Publishing → Add a new pending publisher. Choose GitHub and enter:

| Field | Value |
| --- | --- |
| PyPI project name | `xstars` |
| Owner | `Frankkk1912` |
| Repository name | `xstars` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

On TestPyPI, add the same pending publisher with one difference: Environment name is `testpypi`.

### 3. GitHub environments

In https://github.com/Frankkk1912/xstars/settings/environments create two environments **before** tagging:

- `pypi` — required reviewer: the maintainer (Frankkk1912).
- `testpypi` — required reviewer: the maintainer (Frankkk1912).

The publish jobs wait until that reviewer approves the deployment. Do not add a `PYPI_API_TOKEN` (or any other PyPI token) to the repository or to the environments.

Leave the `testpypi` environment able to deploy from `main` as well as from tags. `workflow_dispatch` runs from a branch and only publishes to TestPyPI. You can limit the `pypi` environment to tags named `v*` if the settings page offers that rule; the workflow already refuses to publish to PyPI unless the event is a `v*` tag.

## Release steps

1. Merge the packaging pull request to `main`. The tagged commit must contain `.github/workflows/publish.yml`.
2. Optional TestPyPI dry run, with no PyPI upload: Actions → **Publish to PyPI** → **Run workflow** on `main`. Approve the `testpypi` environment when GitHub asks. The `pypi` job does not run for `workflow_dispatch`.
3. When the package version in `pyproject.toml` is the version you want to release, tag that commit and push the tag. For this release the tag is `v1.3.1`. The workflow rejects a tag that does not match `version` in `pyproject.toml`.

   ```bash
   git checkout main
   git pull
   git tag v1.3.1
   git push origin v1.3.1
   ```

4. Approve `testpypi`, then approve `pypi`. The order is fixed: PyPI waits until the TestPyPI job succeeds. A repeat upload of the same files to TestPyPI is ignored (`skip-existing: true`). PyPI does not skip an existing version; pick a new version if you need another upload.
5. Check https://pypi.org/project/xstars/ and https://test.pypi.org/project/xstars/ . The PyPI page should point readers at GitHub Releases for the installers.

Do not upload with `twine upload` from a laptop. Trusted publishing is the only path this workflow sets up.

The sdist and wheel must not contain `.xlam`, `.xlsm`, `.xlsx`, `.gif`, `.exe`, or `.pkg`. The Linux CI workflow and `python -m build` plus `twine check --strict` are the checks to re-run before tagging.
