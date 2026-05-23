# Project Site

This folder contains the standalone interactive project introduction page.

Open locally:

```text
project_site/index.html
```

The page is static HTML/CSS/JavaScript, so it does not require a dev server.

For GitHub Pages, set the Pages source to this folder or copy its contents to the configured Pages directory.

## GitHub Pages Deployment

The repository includes a GitHub Actions workflow:

```text
.github/workflows/deploy-project-site.yml
```

When `project_site/**` is pushed to `main`, the workflow uploads this folder and deploys it to GitHub Pages.

In the repository settings, GitHub Pages should use:

```text
Source: GitHub Actions
```
