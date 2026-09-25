# Model Parameters — 175.6M

| Componente | Fórmula | Parámetros |
|---|---|---|
| **Embedding** | 16000 × 768 | 12,288,000 |
| **Por capa (×24)** | | |
| q_proj | 768 × 768 | 589,824 |
| k_proj | 768 × 256 | 196,608 |
| v_proj | 768 × 256 | 196,608 |
| o_proj | 768 × 768 | 589,824 |
| gate/up/down_proj | 768×2048 ×3 | 4,718,592 |
| 2×RMSNorm | 768 ×2 | 1,536 |
| **Subtotal/capa** | | **6,292,992** |
| **24 capas** | ×24 | **151,031,808** |
| **Head** | 768 × 16000 | 12,288,000 |
| Final norm + x0 | | 792 |
| **Total** | | **~175.6M** |
