# Plan: Semantic Heatmap Integration in STAMP

## Context

Der User extrahiert semantische Heatmaps im WSIVL-Repo (via `auto-benchmark` und `direction-benchmark`). Diese Heatmaps erzeugen pro Tile einen Score (Cosine Similarity, normalisiert auf [0,1]), der angibt, wie relevant das Tile fuer eine bestimmte Klasse ist. Die Scores werden als "Semantic Masks" exportiert — entweder direkt in die Quell-H5-Dateien (`/tile/{mask_name}`) oder als `.npy` Sidecar-Dateien.

**Ziel**: Zwei neue Features in STAMP:
1. **Heatmap-gewichtete Features** fuer Crossvalidation-Training (Tile-Level MIL)
2. **Top-K Tile Selection** + Mean-Pooling → Slide-Level Feature → MLP Training (EAGLE-aehnlich)

**Kern-Erkenntnis zur Integration**: WSIVL's `_load_from_h5()` unterstuetzt sowohl `feats`/`coords` (STAMP-Format) als auch `tile/features`/`tile/coords` (WSIVL-Format). Wenn WSIVL's `h5_dir` auf STAMP's Feature-Verzeichnis zeigt, werden die Semantic Masks direkt in die STAMP H5-Dateien geschrieben (`/tile/auto_semantic_mask_{slide_id}`). In diesem Fall sind die Scores bereits 1:1 mit den STAMP-Features aligned — gleicher Index, gleiche Reihenfolge, keine Koordinaten-Konvertierung noetig.

---

## Score-Loading: Drei-Stufen-Strategie

Die Heatmap-Scores koennen aus verschiedenen Quellen kommen. STAMP soll alle drei unterstuetzen:

| Prioritaet | Quelle | Format | Alignment |
|------------|--------|--------|-----------|
| 1 (primaer) | Inline in STAMP H5 | `/tile/{mask_name}` Dataset | Direkt (gleicher Index) |
| 2 | Externe `.npy` Dateien | `{slide_name}.npy` (N,) float32 | Direkt (User garantiert Alignment) |
| 3 | Externe WSIVL H5 Dateien | `/tile/{mask_name}` + `/tile/coords` | Koordinaten-basiert (px→um Konvertierung + Matching) |

### Shared Utility: `src/stamp/modeling/heatmap_scores.py` (NEU)

```python
def load_heatmap_scores(
    feature_h5_path: Path,
    mask_name: str,
    stamp_coords_um: np.ndarray,        # (N, 2)
    stamp_tile_size_um: float,
    stamp_tile_size_px: int | None,
    heatmap_dir: Path | None = None,     # fuer externe Scores
) -> np.ndarray:                          # (N,) float32
    """Laedt Heatmap-Scores aligned zu STAMP Features.
    
    Lade-Reihenfolge:
    1. Inline: /tile/{mask_name}* in der STAMP H5-Datei
    2. Extern .npy: {heatmap_dir}/{slide_name}.npy
    3. Extern H5: {heatmap_dir}/{slide_name}.h5 mit Koordinaten-Alignment
    """
```

Weitere Funktionen:
- `_load_inline_scores(h5_path, mask_name) -> np.ndarray | None`
- `_load_npy_scores(heatmap_dir, slide_name) -> np.ndarray | None`
- `_load_and_align_h5_scores(h5_path, mask_name, stamp_coords_um, mpp) -> np.ndarray`
- `_align_scores_by_coords(stamp_coords_um, wsivl_coords_px, scores, mpp) -> np.ndarray`
  - Wiederverwendung der Logik aus EAGLE's `_align_vir2_to_ctp_by_coords()` (`eagle.py:267-300`)

---

## Feature 1: Heatmap-gewichtete Features fuer Crossvalidation

### Ansatz: BagDataset On-the-fly Weighting

Keine Duplizierung der Feature-Dateien. `BagDataset` laedt optional die Heatmap-Scores und gewichtet Features waehrend des Trainings.

### Gewichtungs-Strategie
```python
weighted_feats[i] = feats[i] * scores[i]   # broadcast (N,1) ueber Feature-Dim D
```
- Scores [0,1]: irrelevante Tiles → ~0, relevante Tiles → unveraendert
- Model-agnostisch: funktioniert mit ViT, TransMIL, MLP

### Aenderungen

#### 1. `src/stamp/modeling/config.py`
Neue Felder in `TrainConfig` (vererbt an `CrossvalConfig`):
```python
heatmap_mask_name: str | None = Field(
    default=None,
    description="Dataset prefix in /tile/ group for heatmap scores (e.g. 'auto_semantic_mask'). Enables feature weighting.",
)
heatmap_dir: Path | None = Field(
    default=None,
    description="Optional: directory with external score files (.npy or .h5)",
)
```

#### 2. `src/stamp/modeling/data.py` — BagDataset erweitern
- `BagDataset.__init__()`: Neue Parameter `heatmap_mask_name: str | None`, `heatmap_dir: Path | None`
- `BagDataset.__getitem__()` (Zeile ~584-655):
  ```python
  # Nach dem Laden von feats und coords_um:
  if self.heatmap_mask_name is not None:
      scores = load_heatmap_scores(
          feature_h5_path=bag_file,
          mask_name=self.heatmap_mask_name,
          stamp_coords_um=coords_um.numpy(),
          stamp_tile_size_um=...,
          stamp_tile_size_px=...,
          heatmap_dir=self.heatmap_dir,
      )
      scores_t = torch.from_numpy(scores).float().unsqueeze(-1)  # (N, 1)
      feats = feats * scores_t  # (N, D) * (N, 1) = (N, D)
  ```
  - Gewichtung passiert VOR dem `_to_fixed_size_bag()` Sampling
- `BagDataset.__getstate__()`: Keine zusaetzlichen Handles zu serialisieren (Scores werden pro Zugriff aus dem gecachten H5-Handle gelesen)

#### 3. `src/stamp/modeling/data.py` — Dataloader-Funktionen
- `tile_bag_dataloader()` (Zeile ~85): Neue Parameter `heatmap_mask_name`, `heatmap_dir` akzeptieren und an `BagDataset` weiterleiten
- `create_dataloader()` (Zeile ~321): Fuer `feature_type == "tile"` durchreichen

#### 4. `src/stamp/modeling/crossval.py`
- `categorical_crossval_()` (Zeile ~48): `config.heatmap_mask_name` und `config.heatmap_dir` an `create_dataloader()` weiterleiten

#### 5. `src/stamp/modeling/train.py`
- `setup_dataloaders_for_training()`: Neue Parameter akzeptieren und weiterleiten
- `train_categorical_model_()`: Aus Config lesen und durchreichen

#### 6. `src/stamp/__main__.py`
- `crossval` und `train` Cases: Keine Aenderung noetig (Felder sind in Config und werden automatisch geladen)

---

## Feature 2: Top-K Tile Selection + Mean-Pooling Encoder

### Ansatz: Neuer Encoder (EAGLE-Pattern)

Kein neuronales Netz noetig — rein score-basierte Selektion + Aggregation. Folgt dem EAGLE-Muster: Top-K Selection → Mean-Pool → Slide-Level H5 → MLP Training.

### Aenderungen

#### 1. `src/stamp/encoding/encoder/heatmap_topk.py` (NEU)

```python
class HeatmapTopK(Encoder):
    """Select top-K tiles by semantic heatmap score, mean-pool to slide embedding.
    
    Like EAGLE but uses pre-computed heatmap scores instead of learned attention.
    No GPU/model needed — purely score-based selection + aggregation.
    """
    
    def __init__(self, k: int = 25, mask_name: str = "auto_semantic_mask"):
        super().__init__(
            model=nn.Identity(),           # Platzhalter, kein NN
            identifier=EncoderName.HEATMAP_TOPK,
            precision=torch.float32,
            required_extractors=[],         # Akzeptiert beliebige Extraktoren
        )
        self.k = k
        self.mask_name = mask_name
    
    def encode_slides_(self, output_dir, feat_dir, device, generate_hash, **kwargs):
        """Laedt Features + Scores, selektiert Top-K, mean-pools zu Slide-Embedding."""
        heatmap_dir = kwargs.get("heatmap_dir")  # Optional externes Score-Verzeichnis
        
        # Output-Verzeichnis erstellen (wie Base-Encoder)
        encode_dir = output_dir / f"{self.identifier}-slide"
        os.makedirs(encode_dir, exist_ok=True)
        
        for h5_file in tqdm(sorted(feat_dir.glob("*.h5"))):
            output_path = (encode_dir / h5_file.stem).with_suffix(".h5")
            if output_path.exists():
                continue
            
            # 1. STAMP Features + Koordinaten laden
            feats, coords_info, extractor = self._read_h5(str(h5_file))
            
            # 2. Heatmap Scores laden (shared utility)
            scores = load_heatmap_scores(
                feature_h5_path=h5_file,
                mask_name=self.mask_name,
                stamp_coords_um=coords_info.coords_um,
                stamp_tile_size_um=coords_info.tile_size_um,
                stamp_tile_size_px=coords_info.tile_size_px,
                heatmap_dir=heatmap_dir,
            )
            
            # 3. Top-K + Mean-Pool
            slide_embedding = self._generate_slide_embedding(
                feats, device, scores=scores
            )
            
            # 4. Speichern als slide-level
            self._save_features_(output_path, slide_embedding, feat_type="slide")
    
    def _generate_slide_embedding(self, feats, device, **kwargs):
        scores = kwargs["scores"]
        scores_t = torch.from_numpy(scores) if isinstance(scores, np.ndarray) else scores
        k = min(self.k, len(scores_t))
        topk_indices = torch.topk(scores_t, k).indices
        topk_feats = feats[topk_indices]
        return topk_feats.mean(dim=0).to(torch.float32).detach().cpu().numpy()
    
    def _generate_patient_embedding(self, feats_list, device, **kwargs):
        scores_list = kwargs.get("scores_list", [])
        all_feats = torch.cat(feats_list, dim=0)
        all_scores = np.concatenate(scores_list)
        return self._generate_slide_embedding(all_feats, device, scores=all_scores)
    
    def encode_patients_(self, output_dir, feat_dir, slide_table_path,
                         patient_label, filename_label, device, generate_hash, **kwargs):
        """Analog zu EAGLE: Pro Patient alle Slides laden, Top-K global, mean-pool."""
        heatmap_dir = kwargs.get("heatmap_dir")
        # ... (folgt EAGLE-Pattern aus eagle.py:188-264)
```

#### 2. `src/stamp/encoding/config.py`
```python
class EncoderName(StrEnum):
    # ... bestehende ...
    HEATMAP_TOPK = "heatmap_topk"

class SlideEncodingConfig(BaseModel):
    # ... bestehende Felder ...
    heatmap_dir: Path | None = None
    heatmap_mask_name: str = "auto_semantic_mask"
    heatmap_top_k: int = 25
```

#### 3. `src/stamp/encoding/__init__.py`
Neue Cases in `init_slide_encoder_()` und `init_patient_encoder_()`:
```python
case EncoderName.HEATMAP_TOPK:
    from stamp.encoding.encoder.heatmap_topk import HeatmapTopK
    selected_encoder = HeatmapTopK(
        k=????,  # muss aus Config kommen
        mask_name=????,
    )
```
**Herausforderung**: `init_slide_encoder_()` bekommt aktuell nur die Parameter aus `SlideEncodingConfig`, nicht die vollen Config-Felder `heatmap_top_k`, `heatmap_mask_name`. Die Funktion muss erweitert werden, um diese Felder durchzureichen (entweder als kwargs oder durch Uebergabe des gesamten Config-Objekts).

Loesung: `heatmap_top_k` und `heatmap_mask_name` als kwargs an `encode_slides_()` durchreichen:
```python
selected_encoder.encode_slides_(
    output_dir=output_dir,
    feat_dir=feat_dir,
    device=device,
    heatmap_dir=heatmap_dir,     # NEU
    generate_hash=generate_hash,
)
```

#### 4. `src/stamp/__main__.py`
Der `encode_slides` Case (Zeile ~104) muss die neuen Felder aus `SlideEncodingConfig` durchreichen:
```python
init_slide_encoder_(
    encoder=config.slide_encoding.encoder,
    output_dir=config.slide_encoding.output_dir,
    feat_dir=config.slide_encoding.feat_dir,
    device=config.slide_encoding.device,
    heatmap_dir=config.slide_encoding.heatmap_dir,        # NEU
    generate_hash=config.slide_encoding.generate_hash,
)
```
Und `init_slide_encoder_()` muss `heatmap_dir`, `heatmap_mask_name`, `heatmap_top_k` als zusaetzliche kwargs akzeptieren.

---

## User-Workflow

### Schritt 1: Features extrahieren (vorhanden)
```bash
stamp preprocess  # → STAMP H5s mit /feats, /coords
```

### Schritt 2: Semantic Heatmaps extrahieren (vorhanden)
WSIVL `auto-benchmark` oder `direction-benchmark` mit `h5_dir` = STAMP Feature-Verzeichnis
→ Schreibt `/tile/auto_semantic_mask_{slide_id}` in STAMP H5-Dateien

### Schritt 3a: Feature 1 — Gewichtetes Crossval-Training
```yaml
# config.yaml
crossval:
  feature_dir: /path/to/features
  heatmap_mask_name: auto_semantic_mask   # Aktiviert Gewichtung
  # heatmap_dir: null                     # Nicht noetig wenn Scores inline
  ...
advanced_config:
  model_name: vit  # oder trans_mil — beliebiges MIL-Modell
  ...
```
```bash
stamp crossval
```

### Schritt 3b: Feature 2 — Top-K Encoder + MLP
```yaml
# config.yaml — Schritt 1: Encoding
slide_encoding:
  encoder: heatmap_topk
  feat_dir: /path/to/features
  output_dir: /path/to/encoded
  heatmap_mask_name: auto_semantic_mask
  heatmap_top_k: 25

# config.yaml — Schritt 2: Training
crossval:
  feature_dir: /path/to/encoded/heatmap_topk-slide
  ...
advanced_config:
  model_name: mlp  # automatisch gewaehlt fuer slide-level
```
```bash
stamp encode_slides   # → Slide-Level H5s
stamp crossval        # → MLP auf Slide-Level Features
```

---

## Implementierungs-Reihenfolge

| Phase | Aufgabe | Dateien |
|-------|---------|---------|
| 1 | Shared Score-Loading Utility | `modeling/heatmap_scores.py` (NEU) |
| 2 | Feature 2: HeatmapTopK Encoder | `encoding/encoder/heatmap_topk.py` (NEU), `encoding/config.py`, `encoding/__init__.py`, `__main__.py` |
| 3 | Feature 1: BagDataset Weighting | `modeling/data.py`, `modeling/config.py`, `modeling/train.py`, `modeling/crossval.py` |
| 4 | Config & Doku | `config.yaml` |

Phase 2 vor Phase 3, weil der Encoder eigenstaendig testbar ist und weniger bestehenden Code aendert.

---

## Alle Datei-Aenderungen

| Datei | Typ | Beschreibung |
|-------|-----|-------------|
| `src/stamp/modeling/heatmap_scores.py` | **NEU** | Shared Utility: Score-Loading mit 3-Stufen-Strategie + Koordinaten-Alignment |
| `src/stamp/encoding/encoder/heatmap_topk.py` | **NEU** | HeatmapTopK Encoder: Top-K Selection + Mean-Pool → Slide-Level |
| `src/stamp/encoding/config.py` | MODIFY | `HEATMAP_TOPK` in EncoderName; `heatmap_dir`, `heatmap_mask_name`, `heatmap_top_k` in SlideEncodingConfig |
| `src/stamp/encoding/__init__.py` | MODIFY | HeatmapTopK Case in `init_slide_encoder_()` und `init_patient_encoder_()` |
| `src/stamp/modeling/config.py` | MODIFY | `heatmap_mask_name`, `heatmap_dir` in TrainConfig |
| `src/stamp/modeling/data.py` | MODIFY | BagDataset: Score-Loading + Gewichtung in `__getitem__()`; Parameter in `tile_bag_dataloader()`, `create_dataloader()` |
| `src/stamp/modeling/train.py` | MODIFY | Score-Parameter durchreichen an Dataloaders |
| `src/stamp/modeling/crossval.py` | MODIFY | Score-Parameter durchreichen an `create_dataloader()` |
| `src/stamp/__main__.py` | MODIFY | Neue Encoding-Parameter durchreichen |
| `src/stamp/config.yaml` | MODIFY | Dokumentation beider Features |

### WSIVL Repo
- Keine Aenderungen noetig wenn WSIVL auf STAMP H5-Dir zeigt (primaerer Pfad)

---

## Verifikation

1. **Unit Tests** fuer `heatmap_scores.py`:
   - Inline Score-Loading aus H5
   - Externe .npy Score-Loading
   - Koordinaten-Alignment (px→um) mit bekannten Werten

2. **Integration Tests**:
   - Erstelle Test-H5 mit Features (10 tiles, D=768) + Inline Scores
   - `HeatmapTopK.encode_slides_()` mit k=3 → verifiziere Output-Shape (D,)
   - `BagDataset` mit Gewichtung → verifiziere `feats * scores`

3. **End-to-End**:
   - `stamp encode_slides` mit `heatmap_topk` auf echte Daten
   - `stamp crossval` mit und ohne `heatmap_mask_name`
   - Metriken vergleichen
