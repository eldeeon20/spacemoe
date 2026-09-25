# Informe Julio

## Resumen general

He revisado los archivos clave de la carpeta `rust/moe-mla`:
- `train.py`
- `model.py`
- `moe.py`
- `mla_attention.py`
- `block.py`
- `moe_block.py`
- `attention.py`
- `rope.py`

La implementación no muestra un error matemático claro en la atención básica ni en la construcción del MoE. Sin embargo, hay puntos importantes que pueden afectar el aprendizaje.

## Hallazgos principales

### 1. Bucle de entrenamiento: paso final con gradientes pendientes

En `train.py`, el bucle principal hace `opt.step()` cuando `micro >= grad_accum`, pero si al final del epoch queda `micro > 0` se ejecuta:

```python
if micro > 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
    opt.step()
    step += 1
```

Problemas potenciales:
- no se llama a `opt.zero_grad()` después de este `opt.step()` final
- `micro` no se reinicia a `0`

Esto puede dejar gradientes residuales acumulados entre epochs y causar actualizaciones incorrectas al final de algunos bloques de datos.

### 2. Configuración de dimensiones en GQA / atención

El modelo usa:
- `d_model = 512`
- `num_heads = 12`
- `head_dim = d_model // num_heads = 42`

Esto produce `num_heads * head_dim = 504`, no `512`.

Consecuencias:
- la proyección Q/K/V trabaja con 504 dimensiones en atención, no con las 512 completas.
- el `o_proj` remapea de 504 a 512.

No es un error de atención directo, pero sí es una configuración no ideal: parte de la dimensión del modelo no se usa de manera uniforme en cada cabeza.

### 3. MoE y balanceo de carga

La lógica de `moe.py` parece coherente:
- router sin bias aprendido
- top-k con normalización de pesos
- capacidad con limitación de tokens
- actualización de `expert_bias`
- z-loss agregado al loss principal

Una observación:
- el balance de carga se actualiza usando `topk_i.flatten()`, por lo que un token con `top_k > 1` puede contar varias veces si ese experto aparece en varias posiciones. Esto es consistente con la forma en que se implementa el ruteo, pero conviene verificar si el comportamiento del `capacity_factor` coincide con la intención.

### 4. Atención MLA / RoPE

La implementación de `mla_attention.py` y `rope.py` es correcta a nivel general:
- Q/KV se divide en componentes de estado y rotación
- las puntuaciones se suman como contenido + RoPE
- se aplica causal mask correctamente
- la función `apply_rope_partial` rota solo la parte seleccionada

No hay errores matemáticos obvios en las fórmulas de atención.

## Posibles causas de que el modelo no aprenda bien

1. **Problema de entrenamiento por gradientes al final de epoch**: el paso final con `micro > 0` puede estar sesgando las actualizaciones.
2. **Arquitectura muy compleja / datos insuficientes**: un modelo con 16 capas, MoE, MLA y un batch efectivo pequeño puede ser difícil de entrenar.
3. **Configuración de `num_heads` y `head_dim`**: aunque no es letal, usar un `d_model` no divisible por `num_heads` crea una atención que no aprovecha todas las dimensiones de forma equilibrada.
4. **Posible hiperparámetro agresivo**: `capacity_factor=1.25`, `top_k=1`, `noise_std=0.01` y `z_loss_gamma=0.001` juntos pueden requerir más cuidado en el ajuste de tasa de aprendizaje y balance.

## Recomendaciones

1. Corregir el bucle de entrenamiento para que después de `opt.step()` siempre se haga `opt.zero_grad()` y `micro = 0`.
2. Usar una combinación de `d_model` y `num_heads` que divida exactamente, por ejemplo `d_model=768` con `num_heads=12` o `d_model=512` con `num_heads=8`.
3. Probar primero sin MoE y/o sin MLA para validar que la base del Transformer aprende.
4. Monitorear `aux_loss` de MoE y la distribución de expertos (`last_counts`) para asegurarse de que el ruteo está equilibrado.

## Conclusión

No hay un error algorítmico evidente en la atención ni en el MoE que invalide completamente el modelo, pero sí hay un fallo en el entrenamiento que puede estar afectando la convergencia. Revisar y corregir el paso final con gradientes pendientes es la primera acción recomendada.
