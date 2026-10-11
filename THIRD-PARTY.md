# Codigo de terceros

Cosxamix se publica bajo la licencia MIT ([LICENSE](LICENSE)). Ademas de sus
dependencias normales, que cargo resuelve y cuyas licencias quedan registradas
en `Cargo.lock`, hay **un archivo copiado** a este repositorio.

## `src/model/qwen2.rs`

Viene de [huggingface/candle](https://github.com/huggingface/candle) 0.9.2,
`candle-transformers/src/models/quantized_qwen2.rs`, distribuido bajo
**MIT OR Apache-2.0** — compatible con la MIT de este proyecto.

Los textos originales estan en
[LICENSE-MIT](https://github.com/huggingface/candle/blob/main/LICENSE-MIT) y
[LICENSE-APACHE](https://github.com/huggingface/candle/blob/main/LICENSE-APACHE)
de ese repositorio.

**Por que esta copiado y no se usa el del crate:** el cache de atencion vive
dentro de `ModelWeights`, es privado, y el tipo no es `Clone`. Con el crate tal
cual hay un solo cache para todo el proceso, asi que dos conversaciones
intercaladas se invalidan mutuamente — medido, 5541 ms y 244 tokens
reaprovechados pasaban a 9770 ms y cero. El razonamiento completo y las dos
alternativas que se descartaron estan en la cabecera del propio archivo y en
[DECISIONES.md](DECISIONES.md) seccion 7.

**Todos los cambios respecto al original estan marcados con `// COSXAMIX:`**,
para que comparar contra upstream siga siendo posible y para que actualizar a
una version nueva de candle sea un trabajo acotado y no una arqueologia.
