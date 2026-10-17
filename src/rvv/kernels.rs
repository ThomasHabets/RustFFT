//! RVV cross-stage kernels. All live vectors stay within a single asm block.
use core::arch::asm;

// Each invocation processes four contiguous complex rows. Segmented accesses deinterleave
// real/imaginary components without assuming any alignment beyond that of the scalar type.
// LMUL=1 leaves enough registers for the butterfly without spilling vectors to memory.
macro_rules! radix4_kernel {
    ($name:ident, $sew:literal, $load:literal, $store:literal, $shift:literal) => {
        /// Safety: V must be supported; data has 4*columns complex values and twiddles has
        /// 3*columns values of the specified precision, columns > 0, and the regions do not alias.
        #[target_feature(enable = "v")]
        #[inline(never)]
        pub(super) unsafe fn $name<const INVERSE: bool>(
            data: *mut u8,
            twiddles: *const u8,
            columns: usize,
        ) {
            let row_bytes = columns << $shift;
            // SAFETY: the caller guarantees all four data rows and three twiddle rows are valid.
            // The loop processes exactly columns values, choosing VL anew for every chunk.
            // All modified vector registers are clobbered. VL/vtype are caller-clobbered under
            // the RISC-V ABI; no Rust code or calls intervene while vector state is live.
            // Memory effects are intentionally unrestricted (no nomem/readonly/pure options).
            asm!(
                "2:",
                concat!("vsetvli {vl}, {remaining}, ", $sew, ", m1, ta, ma"),
                concat!($load, " v0, ({a})"),
                concat!($load, " v2, ({b})"),
                concat!($load, " v8, ({w1})"),
                "vfmul.vv v10, v2, v8",
                "vfnmsac.vv v10, v3, v9",
                "vfmul.vv v11, v3, v8",
                "vfmacc.vv v11, v2, v9",
                "vmv.v.v v2, v10",
                "vmv.v.v v3, v11",
                concat!($load, " v4, ({c})"),
                concat!($load, " v8, ({w2})"),
                "vfmul.vv v10, v4, v8",
                "vfnmsac.vv v10, v5, v9",
                "vfmul.vv v11, v5, v8",
                "vfmacc.vv v11, v4, v9",
                "vmv.v.v v4, v10",
                "vmv.v.v v5, v11",
                concat!($load, " v6, ({d})"),
                concat!($load, " v8, ({w3})"),
                "vfmul.vv v10, v6, v8",
                "vfnmsac.vv v10, v7, v9",
                "vfmul.vv v11, v7, v8",
                "vfmacc.vv v11, v6, v9",
                "vmv.v.v v6, v10",
                "vmv.v.v v7, v11",
                // Even sum/difference and odd sum/difference.
                "vfadd.vv v8, v0, v4",
                "vfadd.vv v9, v1, v5",
                "vfsub.vv v10, v0, v4",
                "vfsub.vv v11, v1, v5",
                "vfadd.vv v12, v2, v6",
                "vfadd.vv v13, v3, v7",
                "vfsub.vv v14, v2, v6",
                "vfsub.vv v15, v3, v7",
                // Conjugate the quarter-turn for inverse transforms, at assembly time.
                ".if {inverse}",
                "vfneg.v v14, v14",
                "vfneg.v v15, v15",
                ".endif",
                "vfadd.vv v0, v8, v12",
                "vfadd.vv v1, v9, v13",
                "vfsub.vv v4, v8, v12",
                "vfsub.vv v5, v9, v13",
                "vfadd.vv v2, v10, v15",
                "vfsub.vv v3, v11, v14",
                "vfsub.vv v6, v10, v15",
                "vfadd.vv v7, v11, v14",
                concat!($store, " v0, ({a})"),
                concat!($store, " v2, ({b})"),
                concat!($store, " v4, ({c})"),
                concat!($store, " v6, ({d})"),
                concat!("slli {step}, {vl}, ", stringify!($shift)),
                "add {a}, {a}, {step}",
                "add {b}, {b}, {step}",
                "add {c}, {c}, {step}",
                "add {d}, {d}, {step}",
                "add {w1}, {w1}, {step}",
                "add {w2}, {w2}, {step}",
                "add {w3}, {w3}, {step}",
                "sub {remaining}, {remaining}, {vl}",
                "bnez {remaining}, 2b",
                a = inout(reg) data => _,
                b = inout(reg) data.add(row_bytes) => _,
                c = inout(reg) data.add(2 * row_bytes) => _,
                d = inout(reg) data.add(3 * row_bytes) => _,
                w1 = inout(reg) twiddles => _,
                w2 = inout(reg) twiddles.add(row_bytes) => _,
                w3 = inout(reg) twiddles.add(2 * row_bytes) => _,
                remaining = inout(reg) columns => _,
                vl = out(reg) _,
                step = out(reg) _,
                inverse = const INVERSE as usize,
                out("v0") _, out("v1") _, out("v2") _, out("v3") _,
                out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                out("v8") _, out("v9") _, out("v10") _, out("v11") _,
                out("v12") _, out("v13") _, out("v14") _, out("v15") _,
                options(nostack),
            );
        }
    };
}

radix4_kernel!(radix4_f32, "e32", "vlseg2e32.v", "vsseg2e32.v", 3);
radix4_kernel!(radix4_f64, "e64", "vlseg2e64.v", "vsseg2e64.v", 4);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        twiddles::{compute_twiddle, rotate_90},
        FftDirection,
    };
    use num_complex::Complex;

    macro_rules! check_kernel {
        ($test:ident, $scalar:ty, $kernel:ident, $tolerance:expr) => {
            #[test]
            fn $test() {
                if !std::arch::is_riscv_feature_detected!("v") { return; }
                for columns in [1, 2, 3, 5, 7, 9, 15, 17, 31, 33, 65] {
                    for direction in [FftDirection::Forward, FftDirection::Inverse] {
                        let guard = Complex::new(123.0 as $scalar, -321.0 as $scalar);
                        let mut data = vec![guard; columns * 4 + 2];
                        for (i, x) in data[1..=4 * columns].iter_mut().enumerate() {
                            *x = Complex::new((i % 13) as $scalar / 13.0, (i % 17) as $scalar / 17.0);
                        }
                        let mut twiddles = vec![guard];
                        for row in 1..4 {
                            for column in 0..columns {
                                twiddles.push(compute_twiddle::<$scalar>(row * column, 4 * columns, direction));
                            }
                        }
                        let mut expected = data.clone();
                        for i in 0..columns {
                            let a = data[1 + i];
                            let b = data[1 + columns + i] * twiddles[1 + i];
                            let c = data[1 + 2 * columns + i] * twiddles[1 + columns + i];
                            let d = data[1 + 3 * columns + i] * twiddles[1 + 2 * columns + i];
                            let even_sum = a + c;
                            let even_diff = a - c;
                            let odd_sum = b + d;
                            let odd_rotated = rotate_90(b - d, direction);
                            expected[1 + i] = even_sum + odd_sum;
                            expected[1 + columns + i] = even_diff + odd_rotated;
                            expected[1 + 2 * columns + i] = even_sum - odd_sum;
                            expected[1 + 3 * columns + i] = even_diff - odd_rotated;
                        }
                        // Offset by one complex value to exercise scalar-aligned addresses.
                        // SAFETY: V was detected above and both allocations hold the required rows.
                        unsafe {
                            if direction == FftDirection::Inverse {
                                $kernel::<true>(data.as_mut_ptr().add(1).cast(), twiddles.as_ptr().add(1).cast(), columns);
                            } else {
                                $kernel::<false>(data.as_mut_ptr().add(1).cast(), twiddles.as_ptr().add(1).cast(), columns);
                            }
                        }
                        assert_eq!(data[0], guard);
                        assert_eq!(*data.last().unwrap(), guard);
                        for (actual, reference) in data.iter().zip(&expected) {
                            assert!(actual.re.is_finite() && actual.im.is_finite());
                            assert!((*actual - reference).norm() < $tolerance,
                                "columns={columns}, direction={direction:?}: {actual:?} != {reference:?}");
                        }
                    }
                }
            }
        };
    }
    check_kernel!(partial_columns_f32, f32, radix4_f32, 1e-6);
    check_kernel!(partial_columns_f64, f64, radix4_f64, 1e-14);
}
