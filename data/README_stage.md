# Stage labels (clinical workflow)

## Files
- `data/stage_taxonomy.json`: stage taxonomy (L1/L2)
- `data/labels.json`: per-paper labels (will be updated with `stage` field)
- `data/stage_meta.json`: derived meta for UI (counts + descriptions)

## How to run
Set API key:
- PowerShell: `setx DEEPSEEK_API_KEY "your_key"`

Then run:
- `F:/202507/AIH_SURVEY/.venv/Scripts/python.exe F:/202507/AIH_SURVEY/label_stages_deepseek.py`

Optional quick test:
- `F:/202507/AIH_SURVEY/.venv/Scripts/python.exe F:/202507/AIH_SURVEY/label_stages_deepseek.py --limit 5`
