# SmartRupee Training Dataset

Expected structure:

```text
dataset/
  train/
    genuine/
    fake/
  validation/
    genuine/
    fake/
  test/
    genuine/
    fake/
```

## Labelling rules

- `genuine`: use only images verified as genuine by a reliable currency-authentication authority or verified source.
- `fake`: use only images of physically verified counterfeit notes or verified counterfeit examples, with clear provenance and appropriate permission.
- Do not infer authenticity from appearance alone.
- Do not generate fake currency images, modify genuine images to simulate counterfeit notes, or invent labels.
- Never place the same physical note or duplicate image in more than one split.
- Keep variants derived from the same physical note together in one split.
- Use consistent note photography and avoid background-only or unrelated images.

Run the read-only audit after adding images:

```powershell
.\venv\Scripts\python.exe prepare_dataset.py
```

The audit reports file counts, empty folders, exact duplicate files, and exact duplicate images appearing in multiple splits.
