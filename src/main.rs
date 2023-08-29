use clap::Parser;
use image::{ImageBuffer, Luma};
use imageproc::map::map_colors;
use ndarray::prelude::*;
use nshare::RefNdarray2;
use pollster::FutureExt as _;
use std::time::Instant;

type GrayscaleImage = ImageBuffer<Luma<u8>, Vec<u8>>;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Implicitly using `std::str::FromStr`
    #[arg(short, long)]
    source: std::path::PathBuf,

    /// Implicitly using `std::str::FromStr`
    #[arg(short, long)]
    target: std::path::PathBuf,

    /// File to which the output image should be written
    #[arg(long, default_value = "result-source.jpg")]
    result_source: String,

    /// File to which the output image should be written
    #[arg(long, default_value = "result-target.jpg")]
    result_target: String,

    /// The number of iterations to find a good approximation
    #[arg(long, default_value_t = 100)]
    iterations: u64,
}

fn approximate_image(
    input: &mut GrayscaleImage,
    target: &mut GrayscaleImage,
    approximator: &dyn compute::BlockApproximator,
) -> (GrayscaleImage, GrayscaleImage) {
    let mut result_source = image::imageops::grayscale(input);
    let mut result_target = image::imageops::grayscale(input);
    let dim = input.dimensions();
    let mut total_error = 0.0;
    println!("image dimensions {:?}", dim);
    for i in 0..(dim.0 / 8) {
        for j in 0..(dim.1 / 8) {
            let input_block = image::imageops::crop(input, i * 8, j * 8, 8, 8).to_image();
            let target_block = image::imageops::crop(target, i * 8, j * 8, 8, 8).to_image();

            let (error, source, target) = approximator
                .approximate(&input_block.ref_ndarray2(), &target_block.ref_ndarray2())
                .block_on();
            total_error += error;

            for n in 0..8 {
                for m in 0..8 {
                    result_source.put_pixel(
                        i * 8 + m,
                        j * 8 + n,
                        image::Luma([source[(n as usize, m as usize)]]),
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
    (result_source, result_target)
}

mod compute;

fn main() {
    env_logger::init();
    let numbers = vec![1, 2, 3, 4];
    //pollster::block_on(compute::run(&numbers));

    let args = Args::parse();
    println!("{args:?}");

    println!("Reading source file: {:?}", args.source);
    let source = image::open(args.source).unwrap();
    let mut source = map_colors(&source, |p| {
        Luma([p[0].saturating_sub(compute::DISTORTION)])
    });

    println!("Reading target file: {:?}", args.target);
    let mut target = image::open(args.target).unwrap().to_luma8();

    if target.dimensions() != source.dimensions() {
        println!("source and target image must have same size");
        return;
    }

    /*if target.dimensions().0 % 8 != 0 || target.dimensions().1 % 8 != 0 {
        println!("Image dimension must be a multiple of 8");
        return;
    }*/

    let now = Instant::now();
    let approximator = compute::Sha512CpuApproximator::new(args.iterations);
    let (result_source, result_target) = approximate_image(&mut source, &mut target, &approximator);

    println!("Writing result source to file: {}", args.result_source);
    result_source.save(args.result_source).unwrap();
    println!("Writing result target to file: {}", args.result_target);
    result_target.save(args.result_target).unwrap();
    println!("Execution time: {}ms", now.elapsed().as_millis());
}
