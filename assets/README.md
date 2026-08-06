# assets/

Exported images for the PawPal+ applied AI system.

The architecture diagram lives as Mermaid source in
[`../diagrams/architecture.mmd`](../diagrams/architecture.mmd), which is the source of
truth. It is embedded directly in the project README as a ```` ```mermaid ```` block —
GitHub renders Mermaid natively, so the diagram displays in the README without needing an
exported image, and it stays readable from source.

## Exporting a PNG (optional)

If you want a raster copy for a slide deck or a PDF:

1. Open [mermaid.live](https://mermaid.live).
2. Paste the contents of `diagrams/architecture.mmd`.
3. Export PNG and save it here as `architecture.png`.

Re-export whenever the `.mmd` changes — the source file is authoritative, and a stale
image is worse than no image.
