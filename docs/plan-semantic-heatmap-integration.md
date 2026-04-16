# Plan: Semantic Heatmap Integration in STAMP

## Context

Der User extrahiert semantische Heatmaps im WSIVL-Repo (via `auto-benchmark` und `direction-benchmark`).
Die Heatmaps werden als **separate H5-Dateien** gespeichert (eine pro Slide) im Verzeichnis
`direction/heatmap_data/` des WSIVL-Benchmark-Outputs.

### Tatsaechliche Heatmap-H5 Struktur (direction-benchmark)

```
{slide_id}.h5
  /x                              # int32 (N,) — Tile X-Koordinaten (aus Quell-H5)
  /y                              # int32 (N,) — Tile Y-Koordinaten (aus Quell-H5)
  /pos                            # float32 (N,) — Cosine Similarity fuer Klasse 1
  /neg                            # float32 (N,) — Cosine Similarity fuer Klasse 2
  /scores/                        # Gruppe mit per-class Scores
    /{class_label_1}              # float32 (N,) — z.B. "HR-proficient"
    /{class_label_2}              # float32 (N,) — z.B. "SET pattern + high TILs + necrosis"
  attrs:
    slide_width                   # int (aus Quell-H5, falls vorhanden)
    slide_height                  # int
    coord_downsample              # int (falls vorhanden)
```

**Wichtig**: Die Scores sind **rohe Cosine-Similarity-Werte** und koennen **negativ** sein
(typischer Wertebereich ca. [-0.3, 0.7]). Sie sind NICHT auf [0,1] normalisiert.

Beispiel-Pfad:
```
/mnt/.../sembench_dir_20260416_012243/run_20260416_032258/direction/heatmap_data/
  17-11003-203_TRF063897.01_A439BL225-001.h5
  17-11003-203_TRF063897.01.h5
  ...
```

Die Koordinaten (`/x`, `/y`) stammen aus den Quell-H5-Dateien via `load_patch_embeddings()`.
Wenn WSIVL's `h5_dir` auf STAMP Feature-Dateien zeigt, sind die Koordinaten in **Mikrometern**
(STAMP-Format). Wenn auf WSIVL-eigene Embeddings, in **Level-0-Pixeln**.

### Daneben existieren auch Inline-Masks (auto-benchmark)

Der auto-benchmark schreibt zusaetzlich normalisierte Masks direkt in die Quell-H5-Dateien:
- `/tile/auto_semantic_mask_{slide_id}` — float32 (N,), [0,1] normalisiert
- Plus `.npy` Sidecar-Dateien

**Ziel**: Zwei neue Features in STAMP:
1. **Heatmap-gewichtete Features** fuer Crossvalidation-Training (Tile-Level MIL)
2. **Top-K Tile Selection** + Mean-Pooling → Slide-Level Feature → MLP Training (EAGLE-aehnlich)

---

## Score-Loading: Primaer externe Heatmap-H5, mit Fallbacks

Die Heatmap-Scores koennen aus verschiedenen Quellen kommen:

| Prioritaet | Quelle | Format | Alignment | Normalisierung |
|------------|--------|--------|-----------|----------------|
| 1 (primaer) | **Direction-Heatmap H5** | `heatmap_dir/{slide}.h5` mit `/scores/{label}`, `/pos`, `/neg`, `/x`, `/y` | Koordinaten-basiert (wenn Einheiten unterschiedlich) oder direkt (wenn gleiche Quelle) | **Noetig**: Rohe Cosine-Scores → [0,1] via minmax |
| 2 | Inline in STAMP H5 | `/tile/{mask_name}` Dataset | Direkt (gleicher Index) | Bereits normalisiert [0,1] |
| 3 | Externe `.npy` Dateien | `{slide_name}.npy` (N,) float32 | Direkt (User garantiert Alignment) | Abhaengig von Quelle |

### Shared Utility: `src/stamp/modeling/heatmap_scores.py` (NEU)

```python
def load_heatmap_scores(
    slide_name: str,
    stamp_coords_um: np.ndarray,        # (N, 2) — STAMP Tile-Koordinaten
    stamp_tile_size_um: float,
    stamp_tile_size_px: int | None,
    heatmap_dir: Path | None = None,     # Verzeichnis mit Heatmap-H5s
    score_key: str = "pos",              # Welcher Score: "pos", "neg", oder Klassen-Label
    feature_h5_path: Path | None = None, # Fuer Inline-Mask Fallback
    normalize: bool = True,              # Scores auf [0,1] normalisieren
) -> np.ndarray:                          # (N,) float32
    """Laedt Heatmap-Scores aligned zu STAMP Features.
    
    Lade-Reihenfolge:
    1. Extern H5: {heatmap_dir}/{slide_name}.h5 — liest /scores/{score_key} oder /{score_key}
    2. Inline: /tile/{score_key}* in der STAMP H5-Datei (feature_h5_path)
    3. Extern .npy: {heatmap_dir}/{slide_name}.npy
    
    Normalisierung (wenn normalize=True):
    - Rohe Cosine-Scores werden per-slide auf [0,1] via minmax normalisiert
    - Bereits normalisierte Inline-Masks werden nicht erneut normalisiert
    """
```

Weitere Funktionen:
- `_load_direction_heatmap_h5(heatmap_dir, slide_name, score_key) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None`
  - Gibt (scores, x_coords, y_coords) zurueck
- `_load_inline_scores(h5_path, mask_name) -> np.ndarray | None`
- `_load_npy_scores(heatmap_dir, slide_name) -> np.ndarray | None`
- `_align_scores_by_coords(stamp_coords, heatmap_x, heatmap_y, scores) -> np.ndarray`
  - Wiederverwendung der Logik aus EAGLE's `_align_vir2_to_ctp_by_coords()` (`eagle.py:267-300`)
  - Wenn Koordinaten bereits uebereinstimmen (gleiche Quelle): direktes Index-Mapping
  - Wenn unterschiedliche Einheiten: Nearest-Neighbor-Matching nach Konvertierung
- `_normalize_scores(scores, method="minmax") -> np.ndarray`
  - `minmax`: `(scores - min) / (max - min)` → [0,1]
  - `sigmoid`: `1 / (1 + exp(-scores))` → (0,1)
  - `raw`: Keine Normalisierung (fuer Top-K Ranking reicht raw)

---

## Feature 1: Heatmap-gewichtete Features fuer Crossvalidation

### Ansatz: BagDataset On-the-fly Weighting

Keine Duplizierung der Feature-Dateien. `BagDataset` laedt optional die Heatmap-Scores
und gewichtet Features waehrend des Trainings.

### Gewichtungs-Strategie
```python
# Scores werden erst auf [0,1] normalisiert (minmax per slide)
normalized = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
weighted_feats[i] = feats[i] * normalized[i]   # broadcast (N,1) ueber Feature-Dim D
```
- Rohe Cosine-Scores (z.B. [-0.2, 0.6]) → normalisiert [0,1]
- Irrelevante Tiles → ~0, relevante Tiles → ~unveraendert
- Model-agnostisch: funktioniert mit ViT, TransMIL, MLP
- Normalisierungs-Methode konfigurierbar (minmax, sigmoid, raw)

### Aenderungen

#### 1. `src/stamp/modeling/config.py`
Neue Felder in `TrainConfig` (vererbt an `CrossvalConfig`):
```python
heatmap_dir: Path | None = Field(
    default=None,
    description="Directory with heatmap H5 files (direction/heatmap_data/). Enables feature weighting.",
)
heatmap_score_key: str = Field(
    default="pos",
    description="Which score to use: 'pos', 'neg', or a class label from /scores/{label}",
)
heatmap_normalize: str = Field(
    default="minmax",
    description="Score normalization: 'minmax' (→[0,1]), 'sigmoid', or 'raw'",
)
```

#### 2. `src/stamp/modeling/data.py` — BagDataset erweitern
- `BagDataset.__init__()`: Neue Parameter `heatmap_dir: Path | None`, `heatmap_score_key: str`, `heatmap_normalize: str`
- `BagDataset.__getitem__()` (Zeile ~584-655):
  ```python
  # Nach dem Laden von feats und coords_um:
  if self.heatmap_dir is not None:
      scores = load_heatmap_scores(
          slide_name=Path(bag_file).stem,
          stamp_coords_um=coords_um.numpy(),
          stamp_tile_size_um=...,
          stamp_tile_size_px=...,
          heatmap_dir=self.heatmap_dir,
          score_key=self.heatmap_score_key,
          feature_h5_path=bag_file,
          normalize=(self.heatmap_normalize != "raw"),
      )
      scores_t = torch.from_numpy(scores).float().unsqueeze(-1)  # (N, 1)
      feats = feats * scores_t  # (N, D) * (N, 1) = (N, D)
  ```
  - Gewichtung passiert VOR dem `_to_fixed_size_bag()` Sampling
- `BagDataset.__getstate__()`: Keine zusaetzlichen Handles zu serialisieren

#### 3. `src/stamp/modeling/data.py` — Dataloader-Funktionen
- `tile_bag_dataloader()` (Zeile ~85): Neue Parameter `heatmap_dir`, `heatmap_score_key`, `heatmap_normalize` akzeptieren und an `BagDataset` weiterleiten
- `create_dataloader()` (Zeile ~321): Fuer `feature_type == "tile"` durchreichen

#### 4. `src/stamp/modeling/crossval.py`
- `categorical_crossval_()` (Zeile ~48): `config.heatmap_dir`, `config.heatmap_score_key`, `config.heatmap_normalize` an `create_dataloader()` weiterleiten

#### 5. `src/stamp/modeling/train.py`
- `setup_dataloaders_for_training()`: Neue Parameter akzeptieren und weiterleiten
- `train_categorical_model_()`: Aus Config lesen und durchreichen

#### 6. `src/stamp/__main__.py`
- `crossval` und `train` Cases: Keine Aenderung noetig (Felder sind in TrainConfig und werden automatisch aus YAML geladen)

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
    
    Heatmap-H5 Struktur (direction-benchmark Output):
      /x              int32 (N,)
      /y              int32 (N,)
      /pos            float32 (N,) — rohe Cosine-Similarity
      /neg            float32 (N,)
      /scores/{label} float32 (N,) — per-class Scores
    
    Scores koennen negativ sein! Fuer Top-K Ranking wird direkt
    auf den rohen Scores gearbeitet (kein minmax noetig).
    """
    
    def __init__(self, k: int = 25, score_key: str = "pos"):
        super().__init__(
            model=nn.Identity(),           # Platzhalter, kein NN
            identifier=EncoderName.HEATMAP_TOPK,
            precision=torch.float32,
            required_extractors=[],         # Akzeptiert beliebige Extraktoren
        )
        self.k = k
        self.score_key = score_key
    
    def encode_slides_(self, output_dir, feat_dir, device, generate_hash, **kwargs):
        """Laedt Features + Scores, selektiert Top-K, mean-pools zu Slide-Embedding."""
        heatmap_dir = kwargs.get("heatmap_dir")
        if not heatmap_dir:
            raise ValueError("heatmap_dir is required for HeatmapTopK encoder")
        
        encode_dir = output_dir / f"{self.identifier}-slide"
        os.makedirs(encode_dir, exist_ok=True)
        
        for h5_file in tqdm(sorted(feat_dir.glob("*.h5"))):
            output_path = (encode_dir / h5_file.stem).with_suffix(".h5")
            if output_path.exists():
                continue
            
            # 1. STAMP Features + Koordinaten laden
            feats, coords_info, extractor = self._read_h5(str(h5_file))
            
            # 2. Heatmap Scores laden (shared utility)
            #    Fuer Top-K: normalize=False (Ranking auf rohen Scores)
            scores = load_heatmap_scores(
                slide_name=h5_file.stem,
                stamp_coords_um=coords_info.coords_um,
                stamp_tile_size_um=coords_info.tile_size_um,
                stamp_tile_size_px=coords_info.tile_size_px,
                heatmap_dir=heatmap_dir,
                score_key=self.score_key,
                feature_h5_path=h5_file,
                normalize=False,  # Top-K braucht keine Normalisierung
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
    heatmap_dir: Path | None = None        # Pfad zu direction/heatmap_data/
    heatmap_score_key: str = "pos"         # "pos", "neg", oder Klassen-Label aus /scores/
    heatmap_top_k: int = 25
```

#### 3. `src/stamp/encoding/__init__.py`
Neue Cases in `init_slide_encoder_()` und `init_patient_encoder_()`:
```python
case EncoderName.HEATMAP_TOPK:
    from stamp.encoding.encoder.heatmap_topk import HeatmapTopK
    selected_encoder = HeatmapTopK(
        k=heatmap_top_k,          # aus kwargs
        score_key=heatmap_score_key,
    )
```

Loesung fuer Parameter-Durchreichung: `init_slide_encoder_()` bekommt neue kwargs
(`heatmap_dir`, `heatmap_score_key`, `heatmap_top_k`), die an `encode_slides_()` weitergereicht
werden. Folgt dem bestehenden Pattern von `agg_feat_dir` bei EAGLE.

```python
def init_slide_encoder_(
    encoder, output_dir, feat_dir, device,
    agg_feat_dir=None, generate_hash=True,
    heatmap_dir=None, heatmap_score_key="pos", heatmap_top_k=25,  # NEU
):
    # ... match encoder ...
    selected_encoder.encode_slides_(
        output_dir=output_dir, feat_dir=feat_dir, device=device,
        heatmap_dir=heatmap_dir, generate_hash=generate_hash,
    )
```

#### 4. `src/stamp/__main__.py`
Der `encode_slides` Case (Zeile ~104) reicht die neuen Felder durch:
```python
init_slide_encoder_(
    encoder=config.slide_encoding.encoder,
    output_dir=config.slide_encoding.output_dir,
    feat_dir=config.slide_encoding.feat_dir,
    device=config.slide_encoding.device,
    agg_feat_dir=config.slide_encoding.agg_feat_dir,
    generate_hash=config.slide_encoding.generate_hash,
    heatmap_dir=config.slide_encoding.heatmap_dir,              # NEU
    heatmap_score_key=config.slide_encoding.heatmap_score_key,  # NEU
    heatmap_top_k=config.slide_encoding.heatmap_top_k,          # NEU
)
```

---

## User-Workflow

### Schritt 1: Features extrahieren (vorhanden)
```bash
stamp preprocess  # → STAMP H5s mit /feats, /coords in Mikrometern
```

### Schritt 2: Semantic Heatmaps extrahieren (vorhanden)
WSIVL `direction-benchmark` erzeugt Heatmap-H5s:
```
sembench_dir_YYYYMMDD/run_YYYYMMDD/direction/heatmap_data/
  {slide_id}.h5   # /x, /y, /pos, /neg, /scores/{class_label}
```

### Schritt 3a: Feature 1 — Gewichtetes Crossval-Training
```yaml
# config.yaml
crossval:
  feature_dir: /path/to/stamp/features
  heatmap_dir: /path/to/sembench_dir/run_.../direction/heatmap_data  # Aktiviert Gewichtung
  heatmap_score_key: pos           # oder "neg", oder Klassen-Label z.B. "HR-proficient"
  heatmap_normalize: minmax        # rohe Cosine-Scores → [0,1]
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
  feat_dir: /path/to/stamp/features
  output_dir: /path/to/encoded
  heatmap_dir: /path/to/sembench_dir/run_.../direction/heatmap_data
  heatmap_score_key: pos           # Score fuer Top-K Ranking
  heatmap_top_k: 25

# config.yaml — Schritt 2: Training
crossval:
  feature_dir: /path/to/encoded/heatmap_topk-slide
  ...
advanced_config:
  model_name: mlp  # automatisch gewaehlt fuer slide-level
```
```bash
stamp encode_slides   # → Slide-Level H5s (mean-pooled Top-K)
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
| `src/stamp/encoding/config.py` | MODIFY | `HEATMAP_TOPK` in EncoderName; `heatmap_dir`, `heatmap_score_key`, `heatmap_top_k` in SlideEncodingConfig |
| `src/stamp/encoding/__init__.py` | MODIFY | HeatmapTopK Case in `init_slide_encoder_()` und `init_patient_encoder_()`; neue kwargs durchreichen |
| `src/stamp/modeling/config.py` | MODIFY | `heatmap_dir`, `heatmap_score_key`, `heatmap_normalize` in TrainConfig |
| `src/stamp/modeling/data.py` | MODIFY | BagDataset: Score-Loading + Gewichtung in `__getitem__()`; Parameter in `tile_bag_dataloader()`, `create_dataloader()` |
| `src/stamp/modeling/train.py` | MODIFY | Score-Parameter durchreichen an Dataloaders |
| `src/stamp/modeling/crossval.py` | MODIFY | Score-Parameter durchreichen an `create_dataloader()` |
| `src/stamp/__main__.py` | MODIFY | Neue Encoding-Parameter durchreichen |
| `src/stamp/config.yaml` | MODIFY | Dokumentation beider Features |

### WSIVL Repo
- Keine Aenderungen noetig — STAMP konsumiert die bestehenden direction/heatmap_data/ H5-Dateien direkt

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
