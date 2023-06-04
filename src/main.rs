use clap::Parser;
use image::{ImageBuffer, Luma};
use imageproc::map::map_colors;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use nshare::RefNdarray2;
use sha2::digest::Update;
use sha2::{Digest, Sha512};
use std::time::Instant;

const N: usize = 8;
const DISTORTION: u8 = 2;

type GrayscaleImage = ImageBuffer<Luma<u8>, Vec<u8>>;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Implicitly using `std::str::FromStr`
    #[arg(short, long)]
    origin: std::path::PathBuf,

    /// Implicitly using `std::str::FromStr`
    #[arg(short, long)]
    target: std::path::PathBuf,

    /// File to which the output image should be written
    #[arg(long, default_value = "result-origin.jpg")]
    result_origin: String,

    /// File to which the output image should be written
    #[arg(long, default_value = "result-target.jpg")]
    result_target: String,

    /// The number of iterations to find a good approximation
    #[arg(long, default_value_t = 100)]
    iterations: u64,
}

trait BlockApproximator {
    fn approximate(
        &self,
        input: &ArrayView2<u8>,
        target: &ArrayView2<u8>,
    ) -> (f32, Array2<u8>, Array2<u8>);
}

struct Sha512CpuApproximator {
    iterations: u64,
}

impl Sha512CpuApproximator {
    fn new(iterations: u64) -> Self {
        Sha512CpuApproximator { iterations }
    }
}

impl BlockApproximator for Sha512CpuApproximator {
    fn approximate(
        &self,
        input: &ArrayView2<u8>,
        target: &ArrayView2<u8>,
    ) -> (f32, Array2<u8>, Array2<u8>) {
        let mut best_origin = Array2::<u8>::zeros((N, N));
        let mut best_target = Array2::<u8>::zeros((N, N));

        let mut error = f32::MAX;

        for _ in 0..self.iterations {
            let delta: Array2<u8> = Array::random((N, N), Uniform::new(0, DISTORTION));
            let current_origin = delta + input;
            let input_vec = current_origin.as_slice().unwrap();
            let mut hasher = Sha512::new();
            Update::update(&mut hasher, input_vec);
            let result = hasher.finalize();

            let current_target = Array::from_iter(result).into_shape((N, N)).unwrap();

            let mut total_error = 0.0;
            for m in 0..N {
                for n in 0..N {
                    let val = target[[m, n]] as f32 - current_target[[m, n]] as f32;
                    total_error += val * val;
                }
            }

            if error > total_error {
                best_origin = current_origin;
                best_target = current_target;
                error = total_error;
            }
        }
        (error, best_origin, best_target)
    }
}

fn approximate_image(
    input: &mut GrayscaleImage,
    target: &mut GrayscaleImage,
    approximator: &dyn BlockApproximator,
) -> (GrayscaleImage, GrayscaleImage) {
    let mut result_origin = image::imageops::grayscale(input);
    let mut result_target = image::imageops::grayscale(input);
    let dim = input.dimensions();
    let mut total_error = 0.0;
    println!("image dimensions {:?}", dim);
    for i in 0..(dim.0 / 8) {
        for j in 0..(dim.1 / 8) {
            let input_block = image::imageops::crop(input, i * 8, j * 8, 8, 8).to_image();
            let target_block = image::imageops::crop(target, i * 8, j * 8, 8, 8).to_image();

            let (error, origin, target) =
                approximator.approximate(&input_block.ref_ndarray2(), &target_block.ref_ndarray2());
            total_error += error;

            for n in 0..8 {
                for m in 0..8 {
                    result_origin.put_pixel(
                        i * 8 + m,
                        j * 8 + n,
                        image::Luma([origin[(n as usize, m as usize)]]),
                    );
                    result_target.put_pixel(
                        i * 8 + m,
                        j * 8 + n,
                        image::Luma([target[(n as usize, m as usize)]]),
                    );
                }
            }
        }
    }
    println!("Total error: {total_error}");
    (result_origin, result_target)
}

fn main() {
    let args = Args::parse();
    println!("{args:?}");

    println!("Reading origin file: {:?}", args.origin);
    let origin = image::open(args.origin).unwrap();
    let mut origin = map_colors(&origin, |p| Luma([p[0].saturating_sub(DISTORTION)]));

    println!("Reading target file: {:?}", args.target);
    let mut target = image::open(args.target).unwrap().to_luma8();

    if target.dimensions() != origin.dimensions() {
        println!("origin and target image must have same size");
        return;
    }

    /*if target.dimensions().0 % 8 != 0 || target.dimensions().1 % 8 != 0 {
        println!("Image dimension must be a multiple of 8");
        return;
    }*/

    let now = Instant::now();
    let approximator = Sha512CpuApproximator::new(args.iterations);
    let (result_origin, result_target) = approximate_image(&mut origin, &mut target, &approximator);

    println!("Writing result origin to file: {}", args.result_origin);
    result_origin.save(args.result_origin).unwrap();
    println!("Writing result target to file: {}", args.result_target);
    result_target.save(args.result_target).unwrap();
    println!("Execution time: {}ms", now.elapsed().as_millis());
}
