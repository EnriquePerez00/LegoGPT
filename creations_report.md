# Reporte Secuencial de Construcciones: model_studs_2_4

Este reporte ilustra cómo se ensamblan paso a paso las 6 creaciones de 2 y 4 piezas generadas por el modelo entrenado, describiendo la posición, orientación y el acoplamiento físico (stud-to-socket) de cada pieza.

---

## 🏗️ Creación #1 (2 Piezas)
*Una placa verde de 2x2 colocada sobre un bloque rectangular celeste de 1x4.*

```
      [ 3022 ] (Plate 2x2 - Verde)
      ====================
      |    |    |    |    |
      --------------------
      [ 3010 ] (Brick 1x4 - Celeste)
```

### Secuencia de Montaje:
1. **Paso 1 (Base)**: Se coloca el bloque **`3010.dat`** (Brick 1x4, Celeste) en la base `(X: 0, Y: 0, Z: 0)`.
2. **Paso 2**: Se monta la placa **`3022.dat`** (Plate 2x2, Verde) en `(X: 10, Y: -24, Z: 20)` rotada 90° alrededor del eje Y. Como `Y = -24`, se acopla directamente sobre los studs superiores en un extremo del bloque base.

---

## 🏗️ Creación #2 (2 Piezas)
*Un bloque marrón de 1x2 colocado debajo de un bloque negro de 1x4.*

```
      [ 3010 ] (Brick 1x4 - Negro)
      --------------------
           |    |
         [ 3004 ] (Brick 1x2 - Marrón)
```

### Secuencia de Montaje:
1. **Paso 1 (Base)**: Se coloca el bloque **`3010.dat`** (Brick 1x4, Negro) en el origen `(X: 0, Y: 0, Z: 0)`.
2. **Paso 2**: Se coloca el bloque **`3004.dat`** (Brick 1x2, Marrón) debajo en `(X: -10, Y: 24, Z: -10)`. Al estar en `Y = 24`, los studs de la pieza 1x2 se insertan en los sockets inferiores del bloque 1x4 negro.

---

## 🏗️ Creación #3 (2 Piezas)
*Un bloque negro de 2x4 montado transversalmente sobre un bloque verde lima de 2x2.*

```
          [ 3001 ] (Brick 2x4 - Negro)
         ====================
              |    |
            [ 3003 ] (Brick 2x2 - Verde Lima)
```

### Secuencia de Montaje:
1. **Paso 1 (Base)**: Se coloca el bloque **`3003.dat`** (Brick 2x2, Verde Lima) en `(X: 0, Y: 0, Z: 0)`.
2. **Paso 2**: Se monta el bloque **`3001.dat`** (Brick 2x4, Negro) transversalmente en `(X: -20, Y: -24, Z: 20)`. Los sockets inferiores del bloque 2x4 se acoplan sobre los studs superiores del brick 2x2 verde.

---

## 🏗️ Creación #4 (4 Piezas)
*Estructura tipo sándwich: Placa roja central con una placa negra superior, una placa blanca inferior y un bloque gris en el tope.*

```
               [ 3010 ] (Brick 1x4 - Gris)
               ---------
             [ 3022 ] (Plate 2x2 - Negro)
             =========
          [ 3020 ] (Plate 2x4 - Rojo)
          =========
        [ 3710 ] (Plate 1x4 - Blanco)
```

### Secuencia de Montaje:
1. **Paso 1 (Base)**: Se coloca la placa **`3020.dat`** (Plate 2x4, Roja) en `(X: 0, Y: 0, Z: 0)`.
2. **Paso 2**: Se acopla la placa **`3022.dat`** (Plate 2x2, Negra) arriba en `(X: 0, Y: -8, Z: 0)`.
3. **Paso 3**: Se coloca la placa **`3710.dat`** (Plate 1x4, Blanca) debajo de la roja en `(X: -10, Y: 8, Z: 0)`.
4. **Paso 4**: Se monta el bloque **`3010.dat`** (Brick 1x4, Gris) arriba en `(X: 20, Y: -16, Z: -10)`.

---

## 🏗️ Creación #5 (2 Piezas)
*Un bloque de 2x2 naranja sostenido en su base por un bloque de 1x4 amarillo.*

```
            [ 3003 ] (Brick 2x2 - Naranja)
            ---------
         [ 3010 ] (Brick 1x4 - Amarillo)
```

### Secuencia de Montaje:
1. **Paso 1 (Base)**: Se sitúa el bloque **`3003.dat`** (Brick 2x2, Naranja) en `(X: 0, Y: 0, Z: 0)`.
2. **Paso 2**: Se coloca el bloque **`3010.dat`** (Brick 1x4, Amarillo) en `(X: 0, Y: 24, Z: 10)`. Los studs del brick amarillo se acoplan a los sockets inferiores del brick naranja.

---

## 🏗️ Creación #6 (4 Piezas)
*Una torre escalonada ascendente de 4 niveles.*

```
                 [ 3022 ] (Plate 2x2 - Amarillo)  (Y = -72)
                 =========
              [ 3003 ] (Brick 2x2 - Rojo)         (Y = -48)
              ---------
           [ 3004 ] (Brick 1x2 - Azul)            (Y = -24)
           ---------
         [ 3005 ] (Brick 1x1 - Verde Lima)        (Y = 0)
```

### Secuencia de Montaje:
1. **Paso 1 (Base)**: Se ubica el bloque base **`3005.dat`** (Brick 1x1, Verde) en `(X: 0, Y: 0, Z: 0)`.
2. **Paso 2**: Se monta el bloque **`3004.dat`** (Brick 1x2, Azul) encima en `(X: 0, Y: -24, Z: 10)`.
3. **Paso 3**: Se monta el bloque **`3003.dat`** (Brick 2x2, Rojo) encima en `(X: 10, Y: -48, Z: 30)`.
4. **Paso 4**: Se corona la torre acoplando la placa **`3022.dat`** (Plate 2x2, Amarilla) encima en `(X: -10, Y: -72, Z: 30)`.
