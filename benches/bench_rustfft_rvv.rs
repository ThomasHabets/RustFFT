mod config;
mod rvv_support;
use criterion::{criterion_group, criterion_main, Criterion};
fn benchmark(c: &mut Criterion) {
    rvv_support::run(c, false);
}
criterion_group! { name = benches; config = config::fast(); targets = benchmark }
criterion_main!(benches);
