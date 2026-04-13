# nl-code

## Headless validation runs

Some dataset validation tasks import plotting libraries such as `matplotlib`.
Run those commands with a non-GUI backend so they do not open windows on the
desktop:

```bash
MPLBACKEND=Agg uv run python ...
```
