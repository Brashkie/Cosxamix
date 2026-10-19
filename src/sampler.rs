//! Elegir el siguiente token a partir de los logits, sin candle.
//!
//! Reproduce la **semántica** del `LogitsProcessor` de candle que usa el
//! motor actual —temperatura, y núcleo de probabilidad acumulada (top-p)—
//! para que los dos motores sean comparables. No reproduce su generador de
//! números: candle usa `StdRng` del crate `rand` y acá va el `Azar` propio,
//! así que con temperatura las secuencias **no** coinciden token a token. Con
//! temperatura cero las dos eligen el máximo y sí coinciden, que es lo que
//! comprueba `compare_forward`.
//!
//! ## Por qué no se ordena el vocabulario entero
//!
//! El núcleo de top-p pide los tokens ordenados por probabilidad, y ordenar
//! 151.936 en cada paso son varios milisegundos contra los ~36 que cuesta el
//! paso entero de decode. Pero el núcleo casi siempre es chico: con la
//! temperatura de un bot la masa se concentra en unas decenas de tokens. Así
//! que primero se separan los `CANDIDATOS` más probables —selección parcial,
//! lineal— y solo esos se ordenan. Si esos no alcanzan la masa pedida, que
//! pasa con distribuciones muy planas, se cae al orden completo: el atajo
//! nunca cambia el resultado, solo lo apura.

use crate::daiss::Azar;

/// Cuántos candidatos se ordenan antes de recurrir al vocabulario entero.
const CANDIDATOS: usize = 1024;

/// Elige tokens con temperatura y top-p.
pub struct Muestreador {
    azar: Azar,
    temperatura: f32,
    top_p: f32,
    /// Buffer de probabilidades, reusado entre pasos: el vocabulario son
    /// 151.936 valores y reservarlos en cada token sería churn sin motivo.
    probs: Vec<f32>,
    orden: Vec<u32>,
}

impl Muestreador {
    /// `temperatura <= 0` significa voraz: siempre el máximo, sin azar.
    pub fn nuevo(semilla: u64, temperatura: f32, top_p: f32) -> Self {
        Self {
            azar: Azar::con_semilla(semilla),
            temperatura,
            top_p: top_p.clamp(0.0, 1.0),
            probs: Vec::new(),
            orden: Vec::new(),
        }
    }

    pub fn voraz() -> Self {
        Self::nuevo(0, 0.0, 1.0)
    }

    /// Un número uniforme en [0, 1).
    fn uniforme(&mut self) -> f32 {
        // 24 bits, que es lo que cabe exacto en la mantisa de un f32.
        (self.azar.siguiente() >> 40) as f32 / 16_777_216.0
    }

    /// El siguiente token.
    pub fn elegir(&mut self, logits: &[f32]) -> u32 {
        if logits.is_empty() {
            return 0;
        }
        if self.temperatura <= 0.0 {
            return argmax(logits);
        }

        // El uniforme se saca ANTES de tomar prestadas las probabilidades: el
        // generador muta y el buffer se presta, y los dos son de `self`.
        let u = self.uniforme();

        // Temperatura y softmax, sobre el buffer reusado.
        self.probs.clear();
        let inv = 1.0 / self.temperatura;
        self.probs.extend(logits.iter().map(|l| l * inv));
        let n = self.probs.len();
        // `softmax` resta el maximo y no puede dar NaN; si `n` es 0 ya se
        // salio arriba.
        let _ = crate::ops::softmax(&mut self.probs, n);

        // Sin nucleo: muestrear de la distribucion entera.
        if self.top_p >= 1.0 {
            let todos: Vec<u32> = (0..n as u32).collect();
            return muestrear_acumulando(&self.probs, todos.iter(), u);
        }

        // El atajo: los CANDIDATOS mas probables, por seleccion parcial.
        self.orden.clear();
        self.orden.extend(0..n as u32);
        let probs = &self.probs;
        let k = CANDIDATOS.min(n);
        if k < n {
            self.orden.select_nth_unstable_by(k - 1, |a, b| {
                probs[*b as usize].total_cmp(&probs[*a as usize])
            });
        }
        let mut nucleo: Vec<u32> = self.orden[..k].to_vec();
        nucleo.sort_unstable_by(|a, b| probs[*b as usize].total_cmp(&probs[*a as usize]));

        let masa: f32 = nucleo.iter().map(|i| probs[*i as usize]).sum();
        if masa < self.top_p && k < n {
            // Distribucion muy plana: los candidatos no alcanzan. Orden
            // completo, que es lo correcto aunque cueste mas.
            nucleo = (0..n as u32).collect();
            nucleo.sort_unstable_by(|a, b| probs[*b as usize].total_cmp(&probs[*a as usize]));
        }

        // Recortar al nucleo: los primeros cuya masa acumulada llega a top_p.
        // El token que cruza el umbral entra, como en candle.
        let mut acumulada = 0f32;
        let mut corte = nucleo.len();
        for (i, t) in nucleo.iter().enumerate() {
            acumulada += probs[*t as usize];
            if acumulada >= self.top_p {
                corte = i + 1;
                break;
            }
        }
        let nucleo = &nucleo[..corte];
        let total: f32 = nucleo.iter().map(|t| probs[*t as usize]).sum();

        // Muestrear dentro del nucleo, renormalizado.
        muestrear_acumulando(probs, nucleo.iter(), u * total)
    }
}

/// El indice del maximo. En empate, el primero, que es lo que hace candle.
pub fn argmax(v: &[f32]) -> u32 {
    let mut mejor = 0usize;
    for (i, x) in v.iter().enumerate() {
        if *x > v[mejor] {
            mejor = i;
        }
    }
    mejor as u32
}

/// Recorre `indices` acumulando `probs` y devuelve el primero que pasa `u`.
fn muestrear_acumulando<'a>(probs: &[f32], indices: impl Iterator<Item = &'a u32>, u: f32) -> u32 {
    let mut acumulada = 0f32;
    let mut ultimo = 0u32;
    for t in indices {
        acumulada += probs[*t as usize];
        ultimo = *t;
        if acumulada > u {
            return *t;
        }
    }
    // Redondeo: si `u` quedo justo en el borde superior, el ultimo.
    ultimo
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Con temperatura cero es el maximo, siempre, y en empate el primero.
    #[test]
    fn voraz_es_el_maximo() {
        let mut m = Muestreador::voraz();
        assert_eq!(m.elegir(&[0.1, 3.0, 2.0, 3.0]), 1);
        assert_eq!(m.elegir(&[-5.0, -1.0, -3.0]), 1);
        assert_eq!(argmax(&[1.0]), 0);
    }

    /// Con top-p chico, solo puede salir el token dominante.
    ///
    /// Un logit de 10 contra el resto en 0 se lleva casi toda la masa; con
    /// top-p de 0,5 el nucleo es solo el. Es la propiedad que distingue un
    /// top-p bien hecho de uno que muestrea de todo el vocabulario.
    #[test]
    fn el_nucleo_recorta_la_cola() {
        let mut logits = vec![0f32; 5000];
        logits[42] = 10.0;
        let mut m = Muestreador::nuevo(7, 1.0, 0.5);
        for _ in 0..200 {
            assert_eq!(m.elegir(&logits), 42);
        }
    }

    /// Las frecuencias tienen que seguir a las probabilidades.
    ///
    /// Tres tokens con probabilidades 0,5 / 0,3 / 0,2 y top-p 1: tras muchas
    /// muestras cada uno tiene que aparecer en su proporcion. Atrapa un
    /// muestreo sesgado, que es el error que no da ningun fallo visible.
    #[test]
    fn las_frecuencias_siguen_a_las_probabilidades() {
        let logits = [0.5f32.ln(), 0.3f32.ln(), 0.2f32.ln()];
        let mut m = Muestreador::nuevo(12345, 1.0, 1.0);
        let n = 30_000;
        let mut cuenta = [0usize; 3];
        for _ in 0..n {
            cuenta[m.elegir(&logits) as usize] += 1;
        }
        for (i, esperado) in [0.5f32, 0.3, 0.2].iter().enumerate() {
            let visto = cuenta[i] as f32 / n as f32;
            assert!(
                (visto - esperado).abs() < 0.015,
                "el token {i} salio {visto} y deberia {esperado}"
            );
        }
    }

    /// Una distribucion plana sobre mas tokens que CANDIDATOS tiene que caer
    /// al orden completo y seguir siendo correcta.
    #[test]
    fn la_distribucion_plana_usa_el_orden_completo() {
        // Todos iguales: el nucleo de 0,9 son el 90% del vocabulario, muy por
        // encima de los 1024 candidatos.
        let logits = vec![1.0f32; 5000];
        let mut m = Muestreador::nuevo(3, 1.0, 0.9);
        let mut vistos = std::collections::HashSet::new();
        for _ in 0..3000 {
            vistos.insert(m.elegir(&logits));
        }
        assert!(
            vistos.len() > 1024,
            "salieron {} tokens distintos: el atajo no cayo al orden completo",
            vistos.len()
        );
    }

    /// La misma semilla da la misma secuencia.
    #[test]
    fn la_semilla_manda() {
        let logits: Vec<f32> = (0..300).map(|i| (i as f32 * 0.07).sin()).collect();
        let a: Vec<u32> = {
            let mut m = Muestreador::nuevo(99, 0.8, 0.9);
            (0..50).map(|_| m.elegir(&logits)).collect()
        };
        let b: Vec<u32> = {
            let mut m = Muestreador::nuevo(99, 0.8, 0.9);
            (0..50).map(|_| m.elegir(&logits)).collect()
        };
        assert_eq!(a, b);
    }
}
