//! El tokenizador propio contra el crate `tokenizers`, token por token.
//!
//! Es lo que permite haberlo sacado del motor: tiene que dar **exactamente** lo
//! mismo, porque un modelo entrenado con un tokenizador no entiende otro. Si
//! esto falla, no es un detalle: el modelo recibe otra cosa.
//!
//! Necesita `modelos/tokenizer.json`, que no va en el repositorio. Sin él, se
//! saltea avisando.

use std::path::Path;

use cosxamix::tokenizer_json::{fin_de_pieza, TokenizadorJson};
use tokenizers::{OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer, Tokenizer};

const RUTA: &str = "modelos/tokenizer.json";

fn cargar() -> Option<(TokenizadorJson, Tokenizer)> {
    if !Path::new(RUTA).exists() {
        eprintln!("sin {RUTA}: no se compara nada");
        return None;
    }
    Some((
        TokenizadorJson::desde_archivo(Path::new(RUTA)).unwrap(),
        Tokenizer::from_file(RUTA).unwrap(),
    ))
}

fn del_crate(tk: &Tokenizer, texto: &str) -> Vec<u32> {
    tk.encode(texto, false).unwrap().get_ids().to_vec()
}

fn comparar(propio: &TokenizadorJson, tk: &Tokenizer, texto: &str) {
    let a = propio.codificar(texto);
    let b = del_crate(tk, texto);
    if a != b {
        let i = a
            .iter()
            .zip(&b)
            .position(|(x, y)| x != y)
            .unwrap_or(a.len().min(b.len()));
        panic!(
            "difieren en el token {i} con {:?}:\n propio {:?}\n crate  {:?}",
            &texto[..texto.len().min(120)],
            &a[i.saturating_sub(2)..(i + 3).min(a.len())],
            &b[i.saturating_sub(2)..(i + 3).min(b.len())],
        );
    }
}

/// Casos elegidos para romper la expresión regular y los especiales.
#[test]
fn los_casos_dificiles() {
    let Some((p, tk)) = cargar() else { return };
    for t in [
        "",
        " ",
        "  ",
        "\n",
        "\r\n",
        " \n ",
        "\n \n",
        "   \n\n   x",
        "a  ",
        "a  b",
        "a \t b",
        "\tx",
        "\u{A0}x",
        "x\u{A0}",
        "x\u{A0}\u{A0}y",
        "\u{2003}\u{2003}",
        "don't DON'T I'LL we've she'd 'S 'ſ 'ſx 'sx 'tx 'REx 'x '",
        "''s '''",
        "123 1,234.5 ١٢٣ Ⅷ ½",
        "hola, chau! ¿sí? ¡no! ...",
        " ... ,, ;; \n",
        "!!\n\n?",
        "año ñandú pingüino Ñ",
        "日本語のテキスト 中文 한국어",
        "emoji 🙂🚀 👩‍💻 y más 🇵🇪",
        "e\u{301} a\u{308} \u{1E9B}\u{323}",
        "\u{212B} \u{FB01} \u{2126}",
        "<|im_start|>system\nHola<|im_end|>\n<|im_start|>user\n¿Qué tal?<|im_end|>\n",
        "<|im_start",
        "<|im_start|",
        "<<|im_end|>>",
        "a<|endoftext|>b<|endoftext|>",
        " <|im_end|> x",
        "<|im_end|>e\u{301}",
        "e<|im_end|>\u{301}",
        "<tool_call>x</tool_call>",
        "\u{0}\u{1}\u{7F}\u{80}\u{FFFD}",
    ] {
        comparar(&p, &tk, t);
    }
}

/// Cargado del compilado, sobre el vocabulario entero: los mismos tokens.
#[test]
fn el_compilado_da_lo_mismo_con_el_vocabulario_de_verdad() {
    let Some((_, tk)) = cargar() else { return };
    // La primera carga lo deja escrito (o ya estaba); estas dos salen de él.
    let dos = TokenizadorJson::desde_archivo(Path::new(RUTA)).unwrap();
    assert!(Path::new("modelos/tokenizer.json.cosxbin").exists());
    comparar(&dos, &tk, "<|im_start|>user\naño 🙂 don't 123<|im_end|>\n");
    let texto = std::fs::read_to_string("README.md").unwrap();
    comparar(&dos, &tk, &texto);
}

/// El vocabulario que viene dentro del GGUF da lo mismo que el `tokenizer.json`:
/// por eso se puede prescindir del JSON.
#[test]
fn el_del_gguf_da_lo_mismo_que_el_json() {
    let Some((propio, tk)) = cargar() else { return };
    let ruta = Path::new("modelos/qwen2.5-3b-instruct-q3_k_m.gguf");
    if !ruta.exists() {
        eprintln!("sin {}: no se compara", ruta.display());
        return;
    }
    let g = cosxamix::gguf::Gguf::abrir(ruta).unwrap();
    let del_gguf = TokenizadorJson::del_gguf(&g).unwrap().expect("lo trae");
    for t in [
        "",
        "<|im_start|>system\nHola<|im_end|>\n<|im_start|>user\n¿Qué tal?<|im_end|>\n",
        "<tool_call>x</tool_call>",
        "don't año 🙂 日本語 123 e\u{301}",
    ] {
        comparar(&del_gguf, &tk, t);
    }
    let texto = std::fs::read_to_string("README.md").unwrap();
    comparar(&del_gguf, &tk, &texto);
    assert_eq!(del_gguf.codificar(&texto), propio.codificar(&texto));
    assert_eq!(
        del_gguf.token_a_id("<|im_end|>"),
        propio.token_a_id("<|im_end|>")
    );

    // Decodificar: también los ids de relleno, que no aportan nada, y los
    // especiales, saltados y sin saltar.
    let ids = del_gguf.codificar("<|im_start|>hola<tool_call>año<|im_end|>");
    let con_relleno: Vec<u32> = ids.iter().copied().chain([151_900, 151_665]).collect();
    for saltar in [true, false] {
        assert_eq!(
            del_gguf.decodificar(&con_relleno, saltar),
            tk.decode(&con_relleno, saltar).unwrap(),
            "saltar = {saltar}"
        );
    }
    // Y un GGUF que no lo trae lo dice, sin error.
    let chico = Path::new("modelos/daiss-mini-q8_0.gguf");
    if chico.exists() {
        let g = cosxamix::gguf::Gguf::abrir(chico).unwrap();
        assert!(TokenizadorJson::del_gguf(&g).unwrap().is_none());
    }
}

/// Decodificar da lo mismo, saltando los especiales y sin saltarlos, también
/// con caracteres cortados a la mitad.
#[test]
fn decodificar_da_lo_mismo() {
    let Some((p, tk)) = cargar() else { return };
    let texto = "<|im_start|>user\naño 🙂 日本語 don't<|im_end|>\n";
    let ids = p.codificar(texto);
    for saltar in [true, false] {
        assert_eq!(
            p.decodificar(&ids, saltar),
            tk.decode(&ids, saltar).unwrap()
        );
        // De a pedazos, como en generación: el pedazo puede cortar un carácter.
        for n in 0..ids.len() {
            assert_eq!(
                p.decodificar(&ids[..n], saltar),
                tk.decode(&ids[..n], saltar).unwrap(),
                "los primeros {n}"
            );
        }
    }
    // Un id que no existe no aporta nada: el modelo tiene 151.936 salidas y el
    // vocabulario, menos.
    assert_eq!(
        p.decodificar(&[151_900, 39_814], true),
        tk.decode(&[151_900, 39_814], true).unwrap()
    );
    assert_eq!(p.token_a_id("<|im_end|>"), tk.token_to_id("<|im_end|>"));
    assert_eq!(p.token_a_id("no existe"), None);
}

/// Los documentos del proyecto enteros: español, código, tablas, símbolos.
#[test]
fn los_documentos_del_proyecto() {
    let Some((p, tk)) = cargar() else { return };
    let mut leidos = 0;
    for entrada in std::fs::read_dir(".").unwrap().flatten() {
        let ruta = entrada.path();
        if ruta.extension().is_some_and(|e| e == "md") {
            let texto = std::fs::read_to_string(&ruta).unwrap();
            comparar(&p, &tk, &texto);
            leidos += 1;
        }
    }
    assert!(leidos > 0);
}

/// **Todo Unicode**, carácter por carácter, en cuatro contextos que distinguen
/// a qué clase pertenece: si un solo carácter cayera en otra clase que en el
/// crate, alguno de estos partiría distinto.
#[test]
fn todo_unicode() {
    let Some((p, tk)) = cargar() else { return };
    let mut t = String::new();
    let mut lote = 0;
    let vaciar = |t: &mut String| {
        comparar(&p, &tk, t);
        t.clear();
    };
    for c in (0..=0x10FFFFu32).filter_map(char::from_u32) {
        // Contextos: pegado a una letra, tras un espacio, entre puntuación y
        // seguido de un espacio. Los separa un salto de línea doble, que corta
        // cualquier pieza.
        t.push('a');
        t.push(c);
        t.push_str("\n\n ");
        t.push(c);
        t.push_str("b\n\n(");
        t.push(c);
        t.push_str(")\n\n");
        t.push(c);
        t.push_str(" \n\n");
        lote += 1;
        if lote == 8192 {
            vaciar(&mut t);
            lote = 0;
        }
    }
    vaciar(&mut t);
}

/// Texto al azar, con todos los pedazos que más se confunden.
#[test]
fn texto_al_azar() {
    let Some((p, tk)) = cargar() else { return };
    const PIEZAS: &[&str] = &[
        "a",
        "Z",
        "ñ",
        "é",
        "e\u{301}",
        "1",
        "٣",
        "½",
        " ",
        "  ",
        "\n",
        "\r\n",
        "\t",
        "\u{A0}",
        "\u{2003}",
        "'",
        "'s",
        "'LL",
        "'ſ",
        ".",
        ",",
        "!",
        "¿",
        "🙂",
        "日",
        "<",
        "|",
        ">",
        "<|im_end|>",
        "<|im_start|>",
        "<|endoftext|>",
        "\u{301}",
        "\u{200D}",
        "hola",
        " mundo",
    ];
    let mut azar = 0x9E37_79B9_7F4A_7C15u64;
    for _ in 0..20_000 {
        let mut t = String::new();
        azar ^= azar << 13;
        azar ^= azar >> 7;
        azar ^= azar << 17;
        for _ in 0..(azar % 24) {
            azar ^= azar << 13;
            azar ^= azar >> 7;
            azar ^= azar << 17;
            t.push_str(PIEZAS[(azar % PIEZAS.len() as u64) as usize]);
        }
        comparar(&p, &tk, &t);
    }
}

// ─── Las piezas, sin pasar por el vocabulario ───────────────────────────────
//
// Comparar tokens no alcanza: si en el vocabulario no hay una fusión que cruce
// un límite, partir ahí o no partir da los mismos tokens. Se comprobó metiendo
// errores a propósito: sacar una letra de la tabla de `\p{L}` pasaba todos los
// tests de arriba. Estos comparan los LÍMITES de cada pieza contra el
// pre-tokenizador del crate, y eso no depende de qué fusiones haya.

fn limites_propios(t: &str) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    let mut i = 0;
    while i < t.len() {
        let f = fin_de_pieza(t, i);
        v.push((i, f));
        i = f;
    }
    v
}

fn limites_del_crate(tk: &Tokenizer, t: &str) -> Vec<(usize, usize)> {
    let mut p = PreTokenizedString::from(t);
    tk.get_pre_tokenizer()
        .unwrap()
        .pre_tokenize(&mut p)
        .unwrap();
    p.get_splits(OffsetReferential::Original, OffsetType::Byte)
        .into_iter()
        .map(|(_, o, _)| o)
        .collect()
}

fn comparar_piezas(tk: &Tokenizer, t: &str) {
    let (a, b) = (limites_propios(t), limites_del_crate(tk, t));
    if a != b {
        let i = a
            .iter()
            .zip(&b)
            .position(|(x, y)| x != y)
            .unwrap_or(a.len().min(b.len()));
        let trozo = |v: &[(usize, usize)]| {
            v.get(i)
                .map(|(d, h)| format!("{:?}", &t[*d..*h]))
                .unwrap_or_default()
        };
        panic!(
            "parten distinto en la pieza {i}: propio {} crate {} (en {:?})",
            trozo(&a),
            trozo(&b),
            t[a.get(i).or(b.get(i)).map_or(0, |x| x.0)..]
                .chars()
                .take(12)
                .collect::<String>()
        );
    }
}

/// Todo Unicode, comparando dónde parte, en los contextos que distinguen a cada
/// alternativa de la expresión regular.
#[test]
fn las_piezas_de_todo_unicode() {
    let Some((_, tk)) = cargar() else { return };
    let mut t = String::new();
    let mut lote = 0;
    for c in (0..=0x10FFFFu32).filter_map(char::from_u32) {
        for contexto in [
            ["a", "b"],  // ¿letra? se pega a las dos
            [" ", "b"],  // ¿va con el espacio y la letra?
            ["(", ")"],  // ¿puntuación? se pega a las dos
            ["1", "2"],  // ¿número? va solo
            ["\n", " "], // ¿espacio en blanco? se junta con los dos
            ["'", "x"],  // ¿contracción?
        ] {
            t.push_str(contexto[0]);
            t.push(c);
            t.push_str(contexto[1]);
            t.push('\u{0}');
        }
        lote += 1;
        if lote == 4096 {
            comparar_piezas(&tk, &t);
            t.clear();
            lote = 0;
        }
    }
    comparar_piezas(&tk, &t);
}

/// La s larga: `(?i)` la pliega a la `s`, y es lo único raro de las contracciones.
#[test]
fn las_piezas_de_las_contracciones() {
    let Some((_, tk)) = cargar() else { return };
    for a in ["'", "x'", " '", "''"] {
        for b in [
            "s", "S", "ſ", "t", "T", "re", "RE", "rE", "Re", "ve", "VE", "m", "M", "ll", "LL",
            "lL", "d", "D", "r", "v", "l", "x", "é", "1", " ", "\n", "K", "\u{212A}",
        ] {
            for c in ["", "x", " ", "é", "1", "'", "\n"] {
                comparar_piezas(&tk, &format!("{a}{b}{c}"));
            }
        }
    }
}

/// Espacios en blanco, en todas las combinaciones chicas: es donde más
/// alternativas compiten y donde hay retroceso.
#[test]
fn las_piezas_de_los_espacios() {
    let Some((_, tk)) = cargar() else { return };
    const P: &[&str] = &[
        " ", "\t", "\n", "\r", "\u{A0}", "\u{2003}", "\u{85}", "a", "1", ".", "\u{301}",
    ];
    let n = P.len();
    for largo in 1..=5u32 {
        for k in 0..(n as u64).pow(largo) {
            let (mut t, mut x) = (String::new(), k);
            for _ in 0..largo {
                t.push_str(P[(x % n as u64) as usize]);
                x /= n as u64;
            }
            comparar_piezas(&tk, &t);
        }
    }
}

/// Un token especial que es prefijo de otro: gana el más largo. En el
/// vocabulario de Qwen no pasa, así que se le agregan.
#[test]
fn entre_especiales_gana_el_mas_largo() {
    if !Path::new(RUTA).exists() {
        return;
    }
    let json = std::fs::read_to_string(RUTA).unwrap();
    let mut v: serde_json::Value = serde_json::from_str(&json).unwrap();
    // Al final y con ids seguidos: el crate renumera los que no vienen en orden.
    let lista = v["added_tokens"].as_array_mut().unwrap();
    let primero = lista.last().unwrap()["id"].as_u64().unwrap() + 1;
    for (k, (texto, especial)) in [("<|a|>", true), ("<|a|>b", false), ("<|a|>bc", true)]
        .into_iter()
        .enumerate()
    {
        lista.push(serde_json::json!({
            "id": primero + k as u64, "content": texto, "single_word": false, "lstrip": false,
            "rstrip": false, "normalized": false, "special": especial,
        }));
    }
    let modificado = v.to_string();

    let propio = TokenizadorJson::desde_json(&modificado).unwrap();
    let ruta = std::env::temp_dir().join("cosxamix-especiales.json");
    std::fs::write(&ruta, &modificado).unwrap();
    let tk = Tokenizer::from_file(&ruta).unwrap();
    let _ = std::fs::remove_file(&ruta);

    for t in [
        "<|a|>",
        "<|a|>b",
        "<|a|>bc",
        "<|a|>bcd",
        "x<|a|>by",
        "<|a|><|a|>b<|a|>bc",
        "<|a|",
    ] {
        comparar(&propio, &tk, t);
        let ids = propio.codificar(t);
        for saltar in [true, false] {
            assert_eq!(
                propio.decodificar(&ids, saltar),
                tk.decode(&ids, saltar).unwrap(),
                "{t}"
            );
        }
    }
    assert_eq!(propio.codificar("<|a|>bc"), [primero as u32 + 2]);
}
