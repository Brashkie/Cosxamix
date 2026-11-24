--  SPDX-License-Identifier: GPL-3.0-or-later

package body Gguf_Bounds with SPARK_Mode is

   procedure Tensor_Span
     (D0, D1, D2, D3               : Unsigned_64;
      Valores_Bloque, Bytes_Bloque : Unsigned_64;
      Inicio, Desde, Largo, Alineacion : Unsigned_64;
      Elementos, Bytes             : out Unsigned_64;
      Ok                           : out Unsigned_8)
   is
      P2, P3, P4, B : Ancho;
      Bloques       : Unsigned_64;
   begin
      Elementos := 0;
      Bytes := 0;
      Ok := 0;

      if not Dims_Ok (D0, D1, D2, D3) then
         return;
      end if;

      --  Cada factor es a lo sumo 2**62 y cada producto parcial que sigue
      --  se comprobo a lo sumo 2**64: ninguna multiplicacion pasa de 2**126.
      P2 := A (D0) * A (D1);
      if P2 > Tope_64 then
         return;
      end if;
      P3 := P2 * A (D2);
      if P3 > Tope_64 then
         return;
      end if;
      P4 := P3 * A (D3);
      if P4 > Tope_64 then
         return;
      end if;

      if Valores_Bloque = 0 or else Bytes_Bloque > Tope_Bloque then
         return;
      end if;
      --  El resto y el cociente en 64 bits: P4 entra.
      if Unsigned_64 (P4) mod Valores_Bloque /= 0 then
         return;
      end if;
      Bloques := Unsigned_64 (P4) / Valores_Bloque;
      --  El cociente de 64 bits es el de 127: los dos son enteros sin signo.
      pragma Assert (A (Unsigned_64 (P4)) = P4);
      pragma Assert (A (Bloques) = P4 / A (Valores_Bloque));
      B := A (Bloques) * A (Bytes_Bloque);

      if Alineacion = 0 or else Desde mod Alineacion /= 0 then
         return;
      end if;
      if A (Inicio) + A (Desde) + B > A (Largo) then
         return;
      end if;

      --  B <= Largo, que es de 64 bits: entra.
      Elementos := Unsigned_64 (P4);
      Bytes := Unsigned_64 (B);
      Ok := 1;
   end Tensor_Span;

end Gguf_Bounds;
