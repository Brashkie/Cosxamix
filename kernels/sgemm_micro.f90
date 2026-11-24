! SPDX-License-Identifier: GPL-3.0-or-later
!
! El microkernel de la GEMM en f32 (src/kernels/sgemm.rs), en Fortran.
!
! Hace lo mismo que `micro_avx2`: un bloque de 6 filas x 16 columnas de C, con
! A y B ya empaquetados en paneles. La diferencia es quien elige las
! instrucciones: en Rust estan escritas a mano con intrinsecos; aca las elige
! gfortran a partir de un bucle comun, que es la promesa de Fortran desde
! siempre y lo que se mide.
!
! Sin runtime de Fortran: `bind(C)`, sin E/S, sin arreglos automaticos ni
! asignables, sin chequeos. El objeto no llama a libgfortran y enlaza igual con
! el linker de MSVC que con el de GNU.
!
! Paneles, en el orden en que los deja `sgemm`:
!   a(6, kc):  a[kk*6 + i]   -> columna kk, fila i
!   b(16, kc): b[kk*16 + j]  -> columna kk, columna de C j
! Resultado en t(16, 6), que en memoria es t[i*16 + j]: la fila i de C seguida,
! como la lee Rust para sumarla en C.

module cosxamix_sgemm
  use, intrinsic :: iso_c_binding, only: c_float, c_size_t
  implicit none
  private
  public :: cosxamix_sgemm_micro_f90, cosxamix_sgemm_micro_suma_f90

contains

  subroutine cosxamix_sgemm_micro_f90(kc, a, b, t) bind(C, name="cosxamix_sgemm_micro_f90")
    integer(c_size_t), value, intent(in) :: kc
    real(c_float), intent(in) :: a(6, kc), b(16, kc)
    real(c_float), intent(out) :: t(16, 6)

    real(c_float) :: acc(16, 6)
    integer(c_size_t) :: kk
    integer :: i

    acc = 0.0_c_float
    do kk = 1, kc
      ! Una columna de B (16 valores, dos registros de 8) contra las 6 filas
      ! de A difundidas: 12 FMA por paso, como el de intrinsecos.
      do i = 1, 6
        acc(:, i) = acc(:, i) + a(i, kk) * b(:, kk)
      end do
    end do
    t = acc
  end subroutine cosxamix_sgemm_micro_f90

  ! Lo mismo, sumado directo en C cuando el bloque esta entero: C es de paso
  ! `ldc` y la fila i de C, c[i*ldc + j], es la columna i de c(ldc, *).
  ! Ahorra la vuelta por `t`, que con `kc` chico (320 en el feed-forward) se
  ! nota.
  subroutine cosxamix_sgemm_micro_suma_f90(kc, a, b, c, ldc) &
      bind(C, name="cosxamix_sgemm_micro_suma_f90")
    integer(c_size_t), value, intent(in) :: kc, ldc
    real(c_float), intent(in) :: a(6, kc), b(16, kc)
    real(c_float), intent(inout) :: c(ldc, 6)

    real(c_float) :: acc(16, 6)
    integer(c_size_t) :: kk
    integer :: i

    acc = 0.0_c_float
    do kk = 1, kc
      do i = 1, 6
        acc(:, i) = acc(:, i) + a(i, kk) * b(:, kk)
      end do
    end do
    do i = 1, 6
      c(1:16, i) = c(1:16, i) + acc(:, i)
    end do
  end subroutine cosxamix_sgemm_micro_suma_f90

end module cosxamix_sgemm
