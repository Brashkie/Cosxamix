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

      --  El contrato, con los argumentos de esta llamada.
      function V return Boolean is
        (Valido (D0, D1, D2, D3, Valores_Bloque, Bytes_Bloque,
                 Inicio, Desde, Largo, Alineacion))
      with Ghost;
   begin
      Elementos := 0;
      Bytes := 0;
      Ok := 0;

      --  Antes de cada salida temprana, la razon por la que Valido es falso:
      --  asi la equivalencia del contrato se demuestra de a pasos chicos, y
      --  la cierran tambien los probadores de gnatprove 14.
      if not Dims_Ok (D0, D1, D2, D3) then
         pragma Assert (not V);
         return;
      end if;

      --  Cada factor es a lo sumo 2**62 y cada producto parcial que sigue
      --  se comprobo a lo sumo 2**64: ninguna multiplicacion pasa de 2**126.
      P2 := A (D0) * A (D1);
      if P2 > Tope_64 then
         pragma Assert (not V);
         return;
      end if;
      P3 := P2 * A (D2);
      if P3 > Tope_64 then
         pragma Assert (not V);
         return;
      end if;
      P4 := P3 * A (D3);
      if P4 > Tope_64 then
         pragma Assert (not V);
         return;
      end if;

      if Valores_Bloque = 0 or else Bytes_Bloque > Tope_Bloque then
         pragma Assert (not V);
         return;
      end if;
      --  El resto y el cociente en 64 bits: P4 entra.
      pragma Assert (A (Unsigned_64 (P4)) = P4);
      if Unsigned_64 (P4) mod Valores_Bloque /= 0 then
         pragma Assert (P4 mod A (Valores_Bloque) /= 0);
         pragma Assert (not V);
         return;
      end if;
      pragma Assert (P4 mod A (Valores_Bloque) = 0);
      Bloques := Unsigned_64 (P4) / Valores_Bloque;
      --  El cociente de 64 bits es el de 127: los dos son enteros sin signo.
      pragma Assert (A (Unsigned_64 (P4)) = P4);
      pragma Assert (A (Bloques) = P4 / A (Valores_Bloque));
      B := A (Bloques) * A (Bytes_Bloque);

      if Alineacion = 0 or else Desde mod Alineacion /= 0 then
         pragma Assert (not V);
         return;
      end if;
      pragma Assert (B = P4 / A (Valores_Bloque) * A (Bytes_Bloque));
      if A (Inicio) + A (Desde) + B > A (Largo) then
         pragma Assert (not V);
         return;
      end if;
      --  Cada condicion de Valido, una por una, antes de juntarlas.
      pragma Assert (Dims_Ok (D0, D1, D2, D3));
      pragma Assert (P2 = E2 (D0, D1));
      pragma Assert (E2 (D0, D1) <= Tope_64);
      pragma Assert (E2 (D0, D1) * A (D2) <= Tope_64);
      pragma Assert (P4 = E2 (D0, D1) * A (D2) * A (D3));
      pragma Assert (E2 (D0, D1) * A (D2) * A (D3) <= Tope_64);
      pragma Assert ((E2 (D0, D1) * A (D2) * A (D3)) mod A (Valores_Bloque) = 0);
      pragma Assert (Desde mod Alineacion = 0);
      pragma Assert
        (A (Inicio) + A (Desde)
         + (E2 (D0, D1) * A (D2) * A (D3)) / A (Valores_Bloque) * A (Bytes_Bloque)
         <= A (Largo));
      pragma Assert (V);

      --  B <= Largo, que es de 64 bits: entra.
      pragma Assert (B <= Tope_64);
      Elementos := Unsigned_64 (P4);
      Bytes := Unsigned_64 (B);
      pragma Assert (A (Elementos) = P4);
      pragma Assert (A (Bytes) = B);
      pragma Assert (A (Bytes) = A (Elementos) / A (Valores_Bloque) * A (Bytes_Bloque));
      Ok := 1;
   end Tensor_Span;

end Gguf_Bounds;
