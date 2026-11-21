# Codigo y modelos de terceros

Cosxamix se publica bajo la **GPL-3.0-or-later** ([LICENSE](LICENSE)). Hasta
la 0.10.0 fue MIT; esas versiones siguen siendo MIT para quien las tenga.

Sus dependencias de Rust las resuelve cargo y sus licencias quedan registradas
en `Cargo.lock`; todas son compatibles con la GPL-3.0 (MIT, Apache-2.0, BSD,
Zlib, Unicode). Desde la 0.11.0 no hay codigo copiado de otro proyecto: el
`src/model/qwen2.rs` que venia de candle salio junto con el motor de candle.

## Modelos que se usan, sin venir en el repositorio

Quien los usa los baja a su carpeta de modelos. La licencia de un modelo es la
de sus pesos, aparte de la del codigo que los corre.

- **Qwen2.5** (huggingface.co/Qwen), el modelo de texto por defecto. Cada
  tamano trae su licencia en su repositorio: el 0,5B, el 1,5B y el 7B son
  Apache-2.0; el **3B es la Qwen Research License**, para investigacion y uso
  no comercial.
- **SD-Turbo** (huggingface.co/stabilityai/sd-turbo), para `/api/image`, bajo la
  **Stability AI Community License**: gratis para uso no comercial; pide mostrar
  **"Powered by Stability AI"** donde se ofrezca, seguir su politica de uso
  (stability.ai/use-policy) y no usar lo que genera para entrenar otro modelo
  generativo. Un uso comercial pide registrarse, y hasta USD 1 millon al ano.
  La licencia completa viene con el modelo (`LICENSE.md`).
- **TAESD** (github.com/madebyollin/taesd), el decodificador de `/api/image`,
  bajo licencia **MIT**.
