--  SPDX-License-Identifier: GPL-3.0-or-later
--
--  Los limites de un tensor de un GGUF, demostrados.
--
--  Un GGUF dice, por cada tensor, sus dimensiones, su formato y donde empieza
--  dentro de la zona de datos. De eso salen cuantos valores tiene y cuantos
--  bytes ocupa, y esos bytes tienen que caer dentro del archivo. Las cuentas
--  son productos y sumas de enteros de 64 bits que vienen del archivo, o sea
--  de quien lo escribio: con dimensiones grandes el producto da la vuelta y
--  queda chico, pasa el chequeo de limites, y despues la cantidad de valores
--  no coincide con los bytes.
--
--  GNATprove demuestra dos cosas de Tensor_Span:
--
--  - que ninguna cuenta se desborda (ausencia de errores de ejecucion);
--  - que dice Ok **exactamente** cuando el tensor es valido segun Valido, que
--    hace las cuentas en aritmetica de 127 bits, donde no hay vuelta; y que
--    entonces los elementos y los bytes son esos numeros.
--
--  Los topes son parte de la regla: cada dimension hasta 2**62 y cada bloque
--  de formato hasta 2**16 bytes. Ningun GGUF real se acerca, y con ellos cada
--  producto entra en 127 bits sin condiciones: el contrato es lineal y los
--  probadores lo cierran solos.
--
--  Pure y sin elaboracion: compilado con -gnatp el objeto no tiene simbolos
--  sin resolver (no hay divisiones de 128 bits, que pedirian la libgcc) y
--  enlaza tambien con el linker de MSVC.

with Interfaces; use Interfaces;

package Gguf_Bounds with SPARK_Mode, Pure is

   type Ancho is range 0 .. 2 ** 127 - 1;

   Tope_Dimension : constant := 2 ** 62;
   Tope_Bloque    : constant := 2 ** 16;
   Tope_64        : constant Ancho := Ancho (Unsigned_64'Last);

   function A (X : Unsigned_64) return Ancho is (Ancho (X));

   --  Las condiciones de un tensor valido, y sus numeros, en aritmetica ancha.
   function Dims_Ok (D0, D1, D2, D3 : Unsigned_64) return Boolean is
     (D0 in 1 .. Tope_Dimension and then D1 in 1 .. Tope_Dimension
      and then D2 in 1 .. Tope_Dimension and then D3 in 1 .. Tope_Dimension);

   function E2 (D0, D1 : Unsigned_64) return Ancho is (A (D0) * A (D1))
   with Pre => D0 <= Tope_Dimension and D1 <= Tope_Dimension;

   function Valido
     (D0, D1, D2, D3               : Unsigned_64;
      Valores_Bloque, Bytes_Bloque : Unsigned_64;
      Inicio, Desde, Largo, Alineacion : Unsigned_64) return Boolean
   is (Dims_Ok (D0, D1, D2, D3)
       and then E2 (D0, D1) <= Tope_64
       and then E2 (D0, D1) * A (D2) <= Tope_64
       and then E2 (D0, D1) * A (D2) * A (D3) <= Tope_64
       and then Valores_Bloque >= 1
       and then Bytes_Bloque <= Tope_Bloque
       and then (E2 (D0, D1) * A (D2) * A (D3)) mod A (Valores_Bloque) = 0
       and then Alineacion >= 1
       and then Desde mod Alineacion = 0
       and then A (Inicio) + A (Desde)
                + (E2 (D0, D1) * A (D2) * A (D3)) / A (Valores_Bloque)
                  * A (Bytes_Bloque)
                <= A (Largo));

   --  Elementos y bytes de un tensor de hasta 4 dimensiones (las que falten
   --  van en 1), y si cae entero dentro de un archivo de `Largo` bytes cuya
   --  zona de datos empieza en `Inicio`, con `Desde` alineado.
   procedure Tensor_Span
     (D0, D1, D2, D3               : Unsigned_64;
      Valores_Bloque, Bytes_Bloque : Unsigned_64;
      Inicio, Desde, Largo, Alineacion : Unsigned_64;
      Elementos, Bytes             : out Unsigned_64;
      Ok                           : out Unsigned_8)
   with
     Export, Convention => C, External_Name => "cosxamix_ada_tensor_span",
     --  La equivalencia va en dos implicaciones y no como `(Ok = 1) =
     --  Valido (...)`: dicen lo mismo, y asi la cierran tambien los
     --  probadores de gnatprove 14, que con la igualdad de booleanos no
     --  podian.
     Post =>
       Ok in 0 | 1
       and then
         (if Ok = 1 then Valido (D0, D1, D2, D3, Valores_Bloque, Bytes_Bloque,
                                 Inicio, Desde, Largo, Alineacion))
       and then
         (if Valido (D0, D1, D2, D3, Valores_Bloque, Bytes_Bloque,
                     Inicio, Desde, Largo, Alineacion)
          then Ok = 1)
       and then
         (if Ok = 1 then
            A (Elementos) = E2 (D0, D1) * A (D2) * A (D3)
            and then A (Bytes) = A (Elementos) / A (Valores_Bloque) * A (Bytes_Bloque)
          else Elementos = 0 and then Bytes = 0);

end Gguf_Bounds;
