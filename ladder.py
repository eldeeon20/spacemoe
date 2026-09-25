"""ladder.py — escalera de capas (intelligence ladders).

Un solo set de pesos sirve para varias profundidades. Cada bloque solo suma al
residual stream, asi que saltar un bloque es la identidad: el modelo queda
simplemente mas chico, en FLOPs, cache y archivo.

Como se eligen los bloques
  Una subred de profundidad d es un conjunto S_d de bloques que se corren en su
  orden original. Los conjuntos tienen que ANIDAR (S2 dentro de S3 dentro de ...
  dentro de S_L) para que un solo set de pesos sirva para todos, y además
  tienen que estar repartidos: un modelo de 2 capas tiene que ver computo
  temprano y tardio, no solo los primeros bloques.

  La regla que da las dos cosas: se arranca con los dos extremos y se parte
  siempre el hueco mas ancho por el medio. Empates van al hueco de la izquierda.

      S_2 = {0, L-1}
      S_{d+1} = S_d union {piso((a+b)/2)},  (a,b) = par adyacente mas separado

  Con eso sale un orden FIJO para cada L, y la subred de profundidad d son los
  primeros d del orden. Para L=16:

      0, 15, 7, 3, 11, 1, 5, 9, 13, 2, 4, 6, 8, 10, 12, 14

  OJO (la trampa): un corte tiene su propia numeracion. Si le recalculas el
  orden a un corte, estas eligiendo bloques que nunca entrenaron juntos como
  subred. Por eso el orden se guarda en el checkpoint y se hereda al cortar.

Un step = una sola profundidad
  En la GPU no se puede "activar un pedazo" de capa: o corre o no corre. La
  granularidad real es el SKIP de capas enteras. Y dentro de un step la
  profundidad es FIJA para todo el batch: si cambiaras la mascara a mitad de un
  step, la cache quedaria mezclando profundidades distintas y revienta. Cambiar
  de profundidad es cambio de step (o de request en inferencia), y ahi se
  reinicia la cache.

  El MoE no se toca: los expertos siguen ruteando por token, normal. Lo unico
  que cambia es cuantas capas corren.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field

# El paper construye la escalera desde 2 capas. Subi este numero si queres un
# piso mas alto (por ejemplo 6) y no regalar la mitad del modelo.
MIN_PROFUNDIDAD = 2

# Proporcion con la que se escala la loss de la subred: d/L.
# El paper usa d/(2L) porque suma CE + lambda*KL en el mismo termino.
ESCALA_LOSS = "d_sobre_L"


def orden_seleccion(n_capas: int) -> list[int]:
    """Orden fijo de activacion para n_capas: extremos y bisection del hueco
    mas ancho. Los primeros d elementos son la subred de profundidad d."""
    if n_capas < 2:
        raise ValueError(f"hacen falta al menos 2 capas, hay {n_capas}")

    elegidos = [0, n_capas - 1]
    orden = list(elegidos)  # orden de activacion: NO se ordena nunca
    while len(elegidos) < n_capas:
        mejor_gap = -1
        mejor_medio = -1
        for i in range(len(elegidos) - 1):
            a, b = elegidos[i], elegidos[i + 1]
            if b - a > mejor_gap:  # empate gana el de la izquierda
                mejor_gap = b - a
                mejor_medio = (a + b) // 2
        elegidos.append(mejor_medio)
        elegidos.sort()
        orden.append(mejor_medio)
    return orden


def indices(n_capas: int, profundidad: int) -> list[int]:
    """Indices (en orden original) de la subred de esa profundidad."""
    if profundidad < MIN_PROFUNDIDAD:
        raise ValueError(
            f"profundidad {profundidad} menor que el piso {MIN_PROFUNDIDAD}")
    if profundidad > n_capas:
        raise ValueError(f"profundidad {profundidad} mayor que {n_capas} capas")
    return sorted(orden_seleccion(n_capas)[:profundidad])


def mascara(n_capas: int, profundidad: int) -> list[bool]:
    """Lista de n_capas bool: True corre, False se saltea."""
    activos = set(indices(n_capas, profundidad))
    return [i in activos for i in range(n_capas)]


def profundidades(n_capas: int, minima: int = MIN_PROFUNDIDAD) -> list[int]:
    """Lospeldaños validos: de minima a n_capas."""
    return list(range(min(2, minima), n_capas + 1))


def elegir_profundidad(
    n_capas: int,
    minima: int = MIN_PROFUNDIDAD,
    rng: random.Random | None = None,
) -> int:
    """Sortea un peldaño para este step. Un step, una profundidad."""
    r = rng or random
    validas = profundidades(n_capas, minima)
    if len(validas) == 1:
        return validas[0]
    return r.choice(validas)


def escala_loss(profundidad: int, n_capas: int) -> float:
    """d/L: la subred aporta en proporcion a la parte del modelo que corrio."""
    return profundidad / float(n_capas)


@dataclass
class Escalera:
    """La escalera de un modelo: cuantas capas tiene y en que orden se activan.

    Guardala en el checkpoint. Si despues cortás el modelo (un slice), el
    corte hereda ESTE orden: recalcularlo sobre la numeracion nueva elegiria
    bloques que nunca entrenaron juntos.
    """

    n_capas: int
    minima: int = MIN_PROFUNDIDAD
    orden: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.orden:
            self.orden = orden_seleccion(self.n_capas)
        elif len(self.orden) != self.n_capas or sorted(self.orden) != list(range(self.n_capas)):
            raise ValueError("el orden tiene que ser una permutacion de las capas")

    def indices(self, profundidad: int) -> list[int]:
        if profundidad < self.minima:
            raise ValueError(f"profundidad {profundidad} menor que el piso {self.minima}")
        if profundidad > self.n_capas:
            raise ValueError(f"profundidad {profundidad} mayor que {self.n_capas}")
        return sorted(self.orden[:profundidad])

    def mascara(self, profundidad: int) -> list[bool]:
        activos = set(self.indices(profundidad))
        return [i in activos for i in range(self.n_capas)]

    def profundidades(self) -> list[int]:
        return list(range(self.minima, self.n_capas + 1))

    def elegir(self, rng: random.Random | None = None) -> int:
        validas = self.profundidades()
        return (rng or random).choice(validas) if len(validas) > 1 else validas[0]

    def guardar(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def cargar(cls, path: str) -> "Escalera":
        with open(path, "r", encoding="utf-8") as f:
            return cls.desde_dict(json.load(f))

    def to_dict(self) -> dict:
        return {"n_capas": self.n_capas, "minima": self.minima, "orden": self.orden}

    @classmethod
    def desde_dict(cls, d: dict) -> "Escalera":
        return cls(n_capas=int(d["n_capas"]), minima=int(d.get("minima", MIN_PROFUNDIDAD)),
                   orden=[int(x) for x in d["orden"]])

    def heredar(self, n_capas_nuevo: int) -> "Escalera":
        """Corta la escalera a un subconjunto de bloques conservando el ORDEN
        del padre, remapeado a la numeracion del corte.

        Importante: el corte se renumera 0..n-1 en orden original, y el orden de
        activacion se traduce con esa tabla. Si en vez de esto se recalculara la
        bisection sobre la numeracion nueva, se estarian eligiendo bloques que
        nunca entrenaron juntos como subred."""
        if n_capas_nuevo > self.n_capas:
            raise ValueError("el corte no puede ser mas grande que el original")
        #

        # Los que sobreviven, en orden original, y su posicion en el corte.
        sobreviven = sorted(self.orden[:n_capas_nuevo])
        posicion = {capa: i for i, capa in enumerate(sobreviven)}
        orden = [posicion[capa] for capa in self.orden[:n_capas_nuevo]]
        return Escalera(n_capas=n_capas_nuevo,
                        minima=min(self.minima, n_capas_nuevo),
                        orden=orden)


def plan_step(
    escalera: Escalera,
    rng: random.Random | None = None,
    p_subred: float = 0.2,
) -> dict:
    """El plan de UN step.

    p_subred = 0.8: modelo completo, loss normal de language modeling.
    p_subred: una subred; en ese step hay que correr la completa como profesor
    (stop-gradient) para el KL contra la subred.

    Devuelve la profundidad, la mascara y el peso de la loss. La mascara es
    FIJA para todo el batch del step: la cache depende de ella.
    """
    r = rng or random
    completa = r.random() < p_subred
    if completa:
        prof = escalera.n_capas
    else:
        prof = escalera.elegir(r)
    return {
        "profundidad": prof,
        "mascara": escalera.mascara(prof),
        "escala": 1.0 if prof == escalera.n_capas else escala_loss(prof, escalera.n_capas),
        "es_profesor": prof != escalera.n_capas,
    }


if __name__ == "__main__":
    import sys

    L = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    esc = Escalera(n_capas=L)
    print(f"capas: {L}  orden: {esc.orden}")
    for d in esc.profundidades():
        act = esc.indices(d)
        print(f"  {d:2d}L -> {act}  ({len(act)}/{L} = {100*len(act)//L}%)")
    peldanios = esc.profundidades()
    print("anidado:",
          all(set(esc.indices(d)) <= set(esc.indices(d + 1))
              for d in peldanios[:-1]))
    print("extremos siempre:",
          all(0 in esc.indices(d) and L - 1 in esc.indices(d) for d in peldanios))
    corte = esc.heredar(8)
    print("corte a 8 hereda el orden:", corte.orden)
    print("mascara 6L:", esc.mascara(6))
