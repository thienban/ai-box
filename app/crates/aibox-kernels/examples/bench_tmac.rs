//! Benchmark autonome (sans dépendance externe) comparant ScalarBackend (KERN-02)
//! vs TmacCpuBackend AVX2 pshufb (KERN-03) sur une projection Bonsai 27B (`5120 x 5120`, M=1).

use std::hint::black_box;
use std::time::Instant;

use aibox_kernels::{
    tensor::{f32_to_f16, pack_ternary_2bit_rows},
    ComputeBackend, TmacCpuBackend, GROUP_SIZE,
};

fn measure_ms<F: FnMut()>(warmup: usize, iters: usize, mut f: F) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed().as_secs_f64() * 1000.0 / (iters as f64)
}

fn main() {
    let m = 1;
    let n = 5120;
    let k = 5120;
    let groups = k / GROUP_SIZE;

    let activations: Vec<f32> = (0..(m * k)).map(|i| ((i as f32) * 0.013).sin()).collect();
    let weights: Vec<i8> = (0..(n * k))
        .map(|i| match (i * 5 + 1) % 3 {
            0 => 0,
            1 => 1,
            _ => -1,
        })
        .collect();
    let scales_fp16 = vec![f32_to_f16(0.05); n * groups];
    let packed = pack_ternary_2bit_rows(&weights, n, k).expect("pack failed");

    let scalar = TmacCpuBackend::scalar_only();
    let avx2 = TmacCpuBackend::new();

    let threads = std::thread::available_parallelism()
        .map(|c| c.get())
        .unwrap_or(1);
    println!("--- Benchmark CPU T-MAC (attn_q 5120x5120, M=1, {threads} threads) ---");
    let scalar_ms = measure_ms(3, 10, || {
        let out = scalar
            .matmul_hadamard_ternary(
                black_box(&activations),
                black_box(&packed),
                black_box(&scales_fp16),
                m,
                n,
                k,
            )
            .unwrap();
        black_box(out);
    });
    println!("ScalarBackend (KERN-02) : {scalar_ms:.3} ms");

    if avx2.uses_avx2() {
        let avx2_ms = measure_ms(3, 10, || {
            let out = avx2
                .matmul_hadamard_ternary(
                    black_box(&activations),
                    black_box(&packed),
                    black_box(&scales_fp16),
                    m,
                    n,
                    k,
                )
                .unwrap();
            black_box(out);
        });
        let speedup = scalar_ms / avx2_ms;
        println!("Avx2Backend   (KERN-03) : {avx2_ms:.3} ms (speedup: {speedup:.2}x)");
    } else {
        println!("AVX2 non disponible sur ce CPU.");
    }
}
