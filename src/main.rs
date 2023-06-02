use image::{DynamicImage, GenericImageView, ImageBuffer, Luma};
use imageproc::map::map_colors;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use nshare::RefNdarray2;
use sha2::digest::Update;
use sha2::{Digest, Sha512};
use std::time::Instant;

const N: usize = 8;
const MAX_ITERATIONS: i32 = 100;
const DISTORTION: u8 = 2;

type GrayscaleImage = ImageBuffer<Luma<u8>, Vec<u8>>;

trait BlockApproximator {
    fn approximate(&self, input: &ArrayView2<u8>, target: &ArrayView2<u8>) -> (f32, Array2<u8>);
}

struct Sha512CpuApproximator;

impl BlockApproximator for Sha512CpuApproximator {
    fn approximate(&self, input: &ArrayView2<u8>, target: &ArrayView2<u8>) -> (f32, Array2<u8>) {
        let mut best_delta = Array2::<u8>::zeros((N, N));

        let mut error = f32::MAX;

        for _ in 0..MAX_ITERATIONS {
            let delta: Array2<u8> = Array::random((N, N), Uniform::new(0, DISTORTION));
            let current_input = delta + input;
            let input_vec = current_input.as_slice().unwrap();
            let mut hasher = Sha512::new();
            Update::update(&mut hasher, input_vec);
            let result = hasher.finalize();

            let a = Array::from_iter(result).into_shape((N, N)).unwrap();

            let mut total_error = 0.0;
            for m in 0..N {
                for n in 0..N {
                    let val = target[[m, n]] as f32 - a[[m, n]] as f32;
                    total_error += val * val;
                }
            }

            if error > total_error {
                best_delta = a; //delta;
                error = total_error;
            }
        }
        (error, best_delta)
    }
}

fn approximate_image(
    input: &mut GrayscaleImage,
    target: &mut GrayscaleImage,
    approximator: &dyn BlockApproximator,
) -> GrayscaleImage {
    let mut result = image::imageops::grayscale(input);
    let dim = input.dimensions();
    let mut total_error = 0.0;
    println!("image dimensions {:?}", dim);
    for i in 0..(dim.0 / 8) {
        for j in 0..(dim.1 / 8) {
            let input_block = image::imageops::crop(input, i * 8, j * 8, 8, 8).to_image();
            let target_block = image::imageops::crop(target, i * 8, j * 8, 8, 8).to_image();

            let (error, delta) =
                approximator.approximate(&input_block.ref_ndarray2(), &target_block.ref_ndarray2());
            total_error += error;

            for n in 0..8 {
                for m in 0..8 {
                    result.put_pixel(
                        i * 8 + n,
                        j * 8 + m,
                        image::Luma([delta[(n as usize, m as usize)]]),
                    );
                }
            }
        }
    }
    println!("Total error: {total_error}");
    result
}

fn main() {
    let mut img = image::open("cat.jpg").unwrap();
    let mut img = map_colors(&img, |p| Luma([p[0].checked_sub(DISTORTION).unwrap_or(0)]));
    let mut target = map_colors(&img, |p| Luma([255 - p[0]]));

    let now = Instant::now();

    let approximator = Sha512CpuApproximator;
    let result = approximate_image(&mut img, &mut target, &approximator);

    result.save("result.png").unwrap();
    println!("Execution time: {}ms", now.elapsed().as_millis());
}
