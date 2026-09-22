use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use sentence_splitter::{Language, Segmenter};

const TEXT: &str = include_str!("data/sample_en.txt");

fn backends(c: &mut Criterion) {
    let punkt = Segmenter::punkt(Language::English).unwrap();
    let icu = Segmenter::icu();
    let auto = Segmenter::auto();

    let mut group = c.benchmark_group("segmentation");
    group.throughput(criterion::Throughput::Bytes(TEXT.len() as u64));
    group.bench_function("punkt", |b| b.iter(|| black_box(punkt.boundaries(TEXT).unwrap())));
    group.bench_function("icu", |b| b.iter(|| black_box(icu.boundaries(TEXT).unwrap())));
    group.bench_function("auto", |b| b.iter(|| black_box(auto.boundaries(TEXT).unwrap())));
    group.finish();
}

criterion_group!(benches, backends);
criterion_main!(benches);
